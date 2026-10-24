import json

import pytest

from fakes import DESCRIPTORS, PROTOCOL, FakeExecutor, FakeWorld
from src.cfg.uqd_presets import resolve_uqd_config
from src.uqd.engine import (CANDIDATE_FAILED, ELIGIBLE, INFRA_UNEVALUATED, BudgetUndefined, UQDEngine)
from src.uqd.loop import initial_generation, run_generation
from src.uqd.seeds import stream_of

TRAIN_KW = dict(timesteps=1000, env_id="Fake-v0", eval_max_steps=100, model_dir="/fake", stats_dir="/fake")


def make_engine(preset, overrides=None, run_dir=None, pop=4, bins=4):
    cfg = resolve_uqd_config(preset, json.dumps(overrides) if overrides else "", population_size=pop)
    return UQDEngine(cfg, DESCRIPTORS, bins, run_seed=7, protocol=PROTOCOL, run_dir=run_dir,
                     initial_population_size=pop)


def child_factory(world, truths_for_children):
    """Children with predetermined truths, named by generation and index."""
    def make(parents, generation):
        kids = []
        for i, _ in enumerate(parents):
            gid = f"g{generation}c{i}"
            world.truths[gid] = truths_for_children(generation, i)
            kids.append(gid)
        return kids
    return make


def test_initial_evaluation_counts_episodes_not_aggregates(tmp_path):
    world = FakeWorld({"a": (100.0, (0.1, 0.1)), "b": (50.0, (0.9, 0.9))})
    engine = make_engine("controlled_baseline", run_dir=str(tmp_path), pop=2)
    snap = initial_generation(engine, FakeExecutor(world), ["a", "b"], TRAIN_KW)
    sid = engine.solution_for_gene("a")
    assert engine.sample_count(sid) == 10                    # ten episodes, ten samples
    sol = engine.solutions[sid]
    assert [s.index for s in sol.rollouts.samples] == list(range(10))
    assert all(stream_of(s.seed) == "search_rollout" for s in sol.rollouts.samples)
    assert stream_of(sol.train_seed) == "train"
    assert snap["illusory_metrics"]["filled_cells"] == 2
    cost = snap["cost"]
    assert cost["initial_train"]["env_steps_measured"] == 2000
    assert cost["initial_eval"]["episodes_measured"] == 20
    for name in ("run_metadata.json",):
        engine.write_run_metadata(".")
        assert (tmp_path / name).exists()
    assert (tmp_path / "archive_gen_0.json").exists() and (tmp_path / "events.jsonl").exists()


def test_reevaluation_continues_seed_stream_and_relocates():
    # a's first 10 episodes look like cell (0,0); its true descriptor is in (3,3)
    world = FakeWorld({"a": (100.0, (0.9, 0.9)), "b": (300.0, (0.9, 0.9))}, descriptor_noise=0.0)
    engine = make_engine("controlled_archive_sampling", pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    sid = engine.solution_for_gene("a")
    first_seeds = engine.solutions[sid].rollouts.seeds()
    world.truths["a"] = (100.0, (0.9, 0.9))
    run_generation(engine, FakeExecutor(world), 1,
                   child_factory(world, lambda g, i: (300.0, (0.9, 0.9))), TRAIN_KW)
    seeds = engine.solutions[sid].rollouts.seeds()
    assert seeds[:10] == first_seeds and seeds[10] == first_seeds[-1] + 1 and len(seeds) == 11
    assert engine.ledger.summary()["reeval"]["episodes_measured"] == 1


def test_archive_sampling_demotes_lucky_solution():
    """A lucky first evaluation is corrected by repeated archive re-evaluation."""
    world = FakeWorld({"lucky": (100.0, (0.1, 0.1)), "solid": (150.0, (0.1, 0.1))},
                      fitness_noise=0.0, descriptor_noise=0.0)
    engine = make_engine("controlled_archive_sampling", pop=2)
    world.truths["lucky"] = (200.0, (0.1, 0.1))         # lucky during its initial evaluation
    initial_generation(engine, FakeExecutor(world), ["lucky", "solid"], TRAIN_KW)
    lucky = engine.solution_for_gene("lucky")
    assert engine.archive.cells[(0, 0)][0] == lucky
    world.truths["lucky"] = (100.0, (0.1, 0.1))         # its true performance
    make = child_factory(world, lambda g, i: (0.0, (0.95, 0.95)))
    for gen in range(1, 25):
        run_generation(engine, FakeExecutor(world), gen, make, TRAIN_KW)
    # Running mean of lucky: (10*200 + n*100) / (10 + n) < 150 once n > 10 re-evaluations
    assert engine.archive.cells[(0, 0)][0] == engine.solution_for_gene("solid")


def test_pas_offspring_samples_follow_archive_median():
    world = FakeWorld({"a": (1.0, (0.1, 0.1)), "b": (2.0, (0.9, 0.9))})
    engine = make_engine("paper_parallel_adaptive_sampling", pop=2)
    engine.cfg["budget"]["sampling_size"] = 64
    assert engine.offspring_samples() == 1               # empty archive -> 1
    initial_generation(engine, FakeExecutor(world), ["a", "b"], TRAIN_KW)
    plan = engine.plan_generation(1)
    assert plan["samples_per_offspring"] == 1             # median count is 1
    assert plan["offspring"] == (64 - 2) // 1             # archive re-evaluation is deducted
    for sid in engine.archive.members():                  # pretend both have 5 samples
        sol = engine.solutions[sid]
        for k in range(4):
            sol.rollouts.add(type(sol.rollouts.samples[0])(seed=10**8 + k + 100 * len(sid), index=1 + k,
                                                           status="ok", fitness=1.0, descriptor=(0.1, 0.1)))
    assert engine.offspring_samples() == 5


def test_sampling_size_budget_undefined_case():
    world = FakeWorld({f"s{i}": (1.0, (i / 4 + 0.01, 0.1)) for i in range(4)})
    engine = make_engine("paper_archive_sampling", pop=4)
    engine.cfg["budget"]["sampling_size"] = 4
    initial_generation(engine, FakeExecutor(world), [f"s{i}" for i in range(4)], TRAIN_KW)
    with pytest.raises(BudgetUndefined):
        engine.plan_generation(1)                          # 4 re-evals leave nothing for offspring


def test_infra_failures_retry_with_same_seeds_then_keep_estimate():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_archive_sampling", pop=1)
    sid_expected = None
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    sid_expected = engine.solution_for_gene("a")
    before = engine.estimate(sid_expected)
    world.infra_fail_plan[sid_expected] = 99              # every re-evaluation is lost
    run_generation(engine, FakeExecutor(world), 1,
                   child_factory(world, lambda g, i: (0.0, (0.9, 0.9))), TRAIN_KW)
    calls = [c for c in world.rollout_calls if c[0] == sid_expected]
    assert len(calls) == 1 + engine.cfg["infra"]["max_retries"]
    assert len({seeds for _, seeds in calls}) == 1        # identical seeds on every retry
    sol = engine.solutions[sid_expected]
    assert sol.lost_reevals == 1 and sol.status == ELIGIBLE
    assert engine.estimate(sid_expected) == before
    assert engine.ledger.summary()["reeval"]["entries_with_unknown_env_steps"] == 3


def test_infra_lost_training_is_not_a_candidate_failure():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_baseline", pop=1)
    sid = None
    world.infra_fail_plan[f"a@t{engine.seeds.train_seed(0)}"] = 99
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    sid = engine.solution_for_gene("a")
    assert engine.solutions[sid].status == INFRA_UNEVALUATED
    assert len(world.train_calls) == 3 and len(set(world.train_calls)) == 1
    assert engine.archive.members() == []


def test_infra_retry_that_succeeds_counts_once():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_baseline", pop=1)
    world.infra_fail_plan[f"a@t{engine.seeds.train_seed(0)}"] = 1
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    sid = engine.solution_for_gene("a")
    assert engine.solutions[sid].status == ELIGIBLE and engine.sample_count(sid) == 10
    summary = engine.ledger.summary()["initial_train"]
    assert summary["by_status"] == {"infra_failure": 1, "ok": 1}


def test_candidate_failures_disqualify_without_inflating():
    world = FakeWorld({"nan": (1e6, (0.1, 0.1)), "bad": (1e6, (0.1, 0.1)), "ok": (5.0, (0.1, 0.1))},
                      nonfinite_genes={"nan"}, build_fail_genes={"bad"})
    engine = make_engine("controlled_baseline", pop=3)
    initial_generation(engine, FakeExecutor(world), ["nan", "bad", "ok"], TRAIN_KW)
    statuses = {engine.solutions[engine.solution_for_gene(g)].status for g in ("nan", "bad")}
    assert statuses == {CANDIDATE_FAILED}
    assert [engine.gene_of(s) for s in engine.archive.members()] == ["ok"]
    assert len(world.train_calls) == 3                     # candidate failures are not retried


def test_candidate_failure_during_reevaluation_removes_archived_policy():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_archive_sampling", pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    world.nonfinite_genes.add("a")
    run_generation(engine, FakeExecutor(world), 1,
                   child_factory(world, lambda g, i: (0.0, (0.9, 0.9))), TRAIN_KW)
    sid = engine.solution_for_gene("a")
    assert engine.solutions[sid].status == CANDIDATE_FAILED and sid not in engine.archive
    assert engine.retired[sid]["reason"].startswith("candidate_failure")


def test_duplicate_offspring_is_not_retrained():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_baseline", pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    assert engine.register_offspring("a", 1, 10) == []


def test_retrain_sampling_counts_training_runs():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("exp_me_sampling_retrain_n3", pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    assert len(world.train_calls) == 3
    assert len({c[1] for c in world.train_calls}) == 3            # three training seeds
    assert all(c[3] == 10 for c in world.train_calls)             # 10 episodes each
    est = engine.estimate("a")
    assert est.n == 3
    summary = engine.solution_summary("a")
    assert len(summary["replicates"]) == 3 and summary["within_run_fitness_sd"] is not None
    assert engine.ledger.summary()["retrain_train"]["jobs_or_requests"] == 2


def test_deep_grid_adds_every_offspring_and_selects_by_roulette():
    world = FakeWorld({f"s{i}": (float(i), (0.1, 0.1)) for i in range(5)})
    engine = make_engine("controlled_deep_grid_sampling_n10", pop=5)
    engine.cfg["depth"] = engine.archive.depth = 3
    initial_generation(engine, FakeExecutor(world), [f"s{i}" for i in range(5)], TRAIN_KW)
    assert len(engine.archive.cells[(0, 0)]) == 3
    assert len(engine.retired) == 2 and all(r["reason"] == "deep_grid_replaced" for r in engine.retired.values())


def test_total_env_step_budget():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_baseline", {"budget": {"total_env_steps": 500}}, pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    assert engine.budget_exhausted()


def test_engine_state_round_trip(tmp_path):
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = make_engine("controlled_archive_sampling", run_dir=str(tmp_path), pop=1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    restored = UQDEngine.load_state(str(tmp_path / "engine_state.pkl"))
    assert restored.archive.members() == engine.archive.members()
    assert restored.rng.bit_generator.state == engine.rng.bit_generator.state


def test_search_rng_reproducible():
    def run():
        world = FakeWorld({f"s{i}": (float(i), (i / 5 + 0.01, 0.5)) for i in range(5)})
        engine = make_engine("controlled_deep_grid_sampling_n10", pop=5)
        initial_generation(engine, FakeExecutor(world), [f"s{i}" for i in range(5)], TRAIN_KW)
        return engine.select_parents(20)
    assert run() == run()
