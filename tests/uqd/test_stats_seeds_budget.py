import math

import pytest

from src.uqd.budget import CostLedger
from src.uqd.seeds import BLOCK_SIZE, STREAM_RANGES, SeedScheme, check_disjoint, stream_of
from src.uqd.stats import (CANDIDATE_FAILURE, INFRA_FAILURE, OK, GeneStats, ReplicateSummary,
                           RolloutSample, RolloutStats, estimate_from_values, outcome_class)


def _sample(seed, f, d=(0.5, 0.5), index=None, status=OK):
    return RolloutSample(seed=seed, index=seed if index is None else index, status=status,
                         fitness=f if status == OK else None, descriptor=d if status == OK else None)


def test_sd_and_sem_are_distinct():
    est = estimate_from_values([1.0, 2.0, 3.0, 4.0], [(0, 0)] * 4)
    assert est.fitness_mean == 2.5
    assert est.fitness_sd == pytest.approx(math.sqrt(5 / 3))          # ddof = 1
    assert est.fitness_sem == pytest.approx(est.fitness_sd / 2.0)     # / sqrt(4)
    single = estimate_from_values([7.0], [(0.1, 0.2)])
    assert single.fitness_sd is None and single.fitness_sem is None and single.descriptor_var is None


def test_descriptor_estimate_is_per_dimension_mean():
    est = estimate_from_values([0, 0], [(0.2, 0.9), (0.4, 0.7)])
    assert est.descriptor_mean == pytest.approx((0.3, 0.8))


def test_rollout_stats_window_and_seed_reuse():
    stats = RolloutStats("ckpt", "proto")
    for k in range(5):
        stats.add(_sample(100 + k, float(k), index=k))
    assert stats.next_index == 5
    assert stats.estimate().fitness_mean == 2.0
    assert stats.estimate(window=2).fitness_mean == 3.5     # most recent two
    with pytest.raises(ValueError, match="already used"):
        stats.add(_sample(100, 1.0, index=9))


def test_samples_reject_nonfinite_and_infra():
    with pytest.raises(ValueError):
        RolloutSample(seed=1, index=0, status=OK, fitness=float("nan"), descriptor=(0, 0))
    with pytest.raises(ValueError, match="infrastructure"):
        RolloutSample(seed=1, index=0, status=INFRA_FAILURE)


def test_candidate_failure_is_flagged_not_averaged():
    stats = RolloutStats("c", "p")
    stats.add(_sample(1, 10.0, index=0))
    stats.add(RolloutSample(seed=2, index=1, status=CANDIDATE_FAILURE, kind="nonfinite_rollout"))
    assert stats.has_candidate_failure
    assert stats.count() == 1


def test_gene_stats_count_each_training_run_once():
    gene = GeneStats("g")
    for r, values in enumerate([[10, 12, 14], [20, 20, 20]]):
        rs = RolloutStats(f"c{r}", "p")
        for k, v in enumerate(values):
            rs.add(_sample(1000 * r + k, float(v), index=k))
        gene.add(ReplicateSummary(train_seed=r, checkpoint_id=f"c{r}", model_path="m", rollouts=rs,
                                  train_timesteps=5))
    est = gene.estimate()
    assert est.n == 2                       # two training runs, not six episodes
    assert est.fitness_mean == pytest.approx(16.0)
    assert gene.within_run_fitness_sd() == pytest.approx(1.0)   # mean of (2, 0)
    with pytest.raises(ValueError):
        gene.add(ReplicateSummary(0, "x", "m", RolloutStats("x", "p"), 5))


def test_outcome_classes():
    assert outcome_class("gene_build_error") == CANDIDATE_FAILURE
    assert outcome_class("nonfinite_rollout") == CANDIDATE_FAILURE
    assert outcome_class("job_lost") == INFRA_FAILURE
    assert outcome_class("fitness_spec_mismatch") == INFRA_FAILURE
    with pytest.raises(ValueError):
        outcome_class("mystery")


def test_stream_ranges_are_disjoint_and_ordered():
    ranges = sorted(STREAM_RANGES.values())
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert a_hi <= b_lo
    assert ranges[-1][1] < 2**31


def test_seed_scheme_streams():
    s0, s1 = SeedScheme(0), SeedScheme(1)
    assert stream_of(s0.train_seed(0)) == "train"
    assert s0.train_seed(0) != s1.train_seed(0)
    roll = s0.rollout_seeds("geneA@t1", 0, 10)
    assert all(stream_of(x) == "search_rollout" for x in roll)
    assert s0.rollout_seeds("geneA@t1", 10, 5) == [roll[0] + 10 + k for k in range(5)]
    assert s0.holdout_seeds(8) == s1.holdout_seeds(8)          # common to every run
    assert all(stream_of(x) == "final_holdout" for x in s0.holdout_seeds(8))
    assert SeedScheme(0, final_master_seed=5).holdout_seeds(4) != s0.holdout_seeds(4)
    with pytest.raises(ValueError, match="exhausted"):
        s0.rollout_seeds("x", BLOCK_SIZE - 1, 2)
    assert stream_of(0) == "legacy"


def test_check_disjoint():
    check_disjoint([1, 2, 3], [4, 5])
    with pytest.raises(ValueError, match="held-out"):
        check_disjoint([1, 2, 3], [3, 4])


def test_ledger_separates_measured_and_unknown():
    ledger = CostLedger()
    ledger.record("initial_train", "a", 0, env_steps=1000, wall_seconds=5.0)
    ledger.record("initial_eval", "a", 0, env_steps=50, episodes=10, wall_seconds=1.0)
    ledger.record("reeval", "a", 1, env_steps=None, status="infra_failure", note="job_lost")
    ledger.record("final_eval", "a", 2, env_steps=999, episodes=1)
    summary = ledger.summary()
    assert ledger.search_env_steps() == 1050           # final eval is not search cost
    assert summary["reeval"]["entries_with_unknown_env_steps"] == 1
    assert summary["reeval"]["by_status"] == {"infra_failure": 1}
    restored = CostLedger.from_list(ledger.to_list())
    assert restored.summary() == summary
    with pytest.raises(ValueError):
        ledger.record("nonsense")


def test_search_stream_is_namespaced_by_preset_but_training_is_shared():
    a, b = SeedScheme(0, search_namespace="controlled_baseline"), SeedScheme(0, search_namespace="paper_deep_grid")
    assert a.search_seed() != b.search_seed()          # different gene ids in concurrent runs
    assert a.train_seed(0) == b.train_seed(0)          # same training seeds across presets
    assert a.rollout_seeds("g@t1", 0, 3) == b.rollout_seeds("g@t1", 0, 3)
