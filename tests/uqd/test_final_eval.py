import json

import pytest

from fakes import DESCRIPTORS, PROTOCOL, FakeExecutor, FakeWorld
from src.cfg.uqd_presets import resolve_uqd_config
from src.uqd.engine import UQDEngine
from src.uqd.final_eval import compute_final_metrics, episode_stats, final_requests
from src.uqd.archive import Grid
from src.uqd.loop import initial_generation
from src.uqd.seeds import check_disjoint

TRAIN_KW = dict(timesteps=1000, env_id="Fake-v0", eval_max_steps=100, model_dir="/f", stats_dir="/f")


def _run_final(engine, world, m=16):
    snap = engine.snapshot()
    seeds = engine.seeds.holdout_seeds(m)
    check_disjoint(engine.search_seeds_used(), seeds)
    reqs = final_requests(snap, seeds)
    results = FakeExecutor(world).run_rollouts(reqs, PROTOCOL)
    episodes = {r["solution_id"]: r["episodes"] for r in results}
    return snap, compute_final_metrics(snap, episodes, engine.cfg["final_eval"], seeds)


def _engine(preset, pop):
    cfg = resolve_uqd_config(preset, "", population_size=pop)
    return UQDEngine(cfg, DESCRIPTORS, 4, run_seed=3, protocol=PROTOCOL, initial_population_size=pop)


def test_lucky_descriptor_is_corrected():
    # "drifter" was evaluated in (0,0) but truly lives in (3,3) where "home" is better
    world = FakeWorld({"drifter": (100.0, (0.1, 0.1)), "home": (200.0, (0.9, 0.9))},
                      fitness_noise=1.0, descriptor_noise=0.0)
    engine = _engine("controlled_baseline", 2)
    initial_generation(engine, FakeExecutor(world), ["drifter", "home"], TRAIN_KW)
    world.truths["drifter"] = (100.0, (0.9, 0.9))
    snap, metrics = _run_final(engine, world)
    best = metrics["reports"]["best_only"]
    assert metrics["illusory"]["filled_cells"] == 2
    assert best["filled_cells"] == 1                       # both land in (3,3); best kept
    assert best["correct_cell_fraction"] == pytest.approx(0.5)
    assert best["qd_score_loss"] > 0
    kept = best["corrected_archive"]["3,3"]
    assert kept["solution"] == engine.solution_for_gene("home")


def test_episode_stats_reliability_fields():
    grid = Grid(DESCRIPTORS, 4)
    eps = [{"return": float(r), "length": 10, "nonfinite": False,
            "descriptor": {"foot_contact_0": d, "foot_contact_1": 0.1}}
           for r, d in [(1, 0.1), (2, 0.1), (3, 0.3), (10, 0.1)]]
    st = episode_stats(eps, grid, (0, 0))
    assert st["fitness_estimate"] == 2.5 and st["fitness_mean"] == 4.0   # median vs mean
    assert st["p_in_corrected_cell"] == 0.75 and st["p_in_search_cell"] == 0.75
    assert st["ndv"] < 0
    bad = episode_stats(eps + [{"nonfinite": True}], grid, (0, 0))
    assert bad["valid"] is False and bad["nonfinite_episodes"] == 1


def test_heldout_nonfinite_is_reported_not_hidden():
    world = FakeWorld({"a": (10.0, (0.1, 0.1)), "b": (20.0, (0.9, 0.9))})
    engine = _engine("controlled_baseline", 2)
    initial_generation(engine, FakeExecutor(world), ["a", "b"], TRAIN_KW)
    world.nonfinite_genes.add("a")
    _, metrics = _run_final(engine, world)
    best = metrics["reports"]["best_only"]
    assert best["heldout_candidate_failures"] == 1 and best["filled_cells"] == 1


def test_deep_grid_in_cell_selector_report():
    world = FakeWorld({f"s{i}": (float(10 * i), (0.1, 0.1)) for i in range(3)})
    engine = _engine("controlled_deep_grid_sampling_n10", 3)
    initial_generation(engine, FakeExecutor(world), ["s0", "s1", "s2"], TRAIN_KW)
    _, metrics = _run_final(engine, world, m=32)
    ics = metrics["entities"]["in_cell_selector"][0]
    assert sum(ics["member_draw_counts"]) == 32
    # the worst member (lowest search estimate) has roulette weight 0
    worst = min(range(3), key=lambda i: engine.fitness_of(ics["members"][i]))
    assert ics["member_draw_counts"][worst] == 0
    assert set(metrics["reports"]) == {"best_only", "in_cell_selector"}


def test_retrain_gene_final_eval_uses_every_replicate():
    world = FakeWorld({"a": (10.0, (0.1, 0.1))})
    engine = _engine("exp_me_sampling_retrain_n3", 1)
    initial_generation(engine, FakeExecutor(world), ["a"], TRAIN_KW)
    snap = engine.snapshot()
    reqs = final_requests(snap, engine.seeds.holdout_seeds(4))
    assert len(reqs) == 3
    _, metrics = _run_final(engine, world, m=4)
    ent = metrics["entities"]["best_only"][0]
    assert len(ent["replicates"]) == 3 and ent["valid"]
