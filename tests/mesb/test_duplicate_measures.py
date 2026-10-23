"""Duplicate descriptor values and tiny populations must not crash."""

import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive


def test_all_identical_measures(rec):
    a = SlidingBoundariesArchive(dims=(5, 5), remap_frequency=10, seed=0)
    a.initialize([rec(f"g{i}", float(i), (1.0, 2.0)) for i in range(8)])
    assert a.boundaries == [[1.0] * 6, [2.0] * 6]
    assert a.num_elites == 1 and a.best_elite().gene_id == "g7"
    assert a.degenerate_dimensions() == [0, 1]
    a.add_many([rec(f"h{i}", 0.0, (1.0, 2.0)) for i in range(20)])  # two remaps
    assert a.num_remaps == 2 and a.num_elites == 1


def test_fewer_unique_values_than_bins(rec):
    a = SlidingBoundariesArchive(dims=(10,), seed=0)
    a.initialize([rec("a", 1.0, (0.0,)), rec("b", 2.0, (5.0,)), rec("c", 3.0, (5.0,))])
    edges = a.boundaries[0]
    assert len(edges) == 11 and edges[0] == 0.0 and edges[-1] == 5.0
    assert a.num_elites == 2  # distinct values -> distinct reachable cells


def test_single_initial_individual(rec):
    a = SlidingBoundariesArchive(dims=(4, 4), seed=0)
    a.initialize([rec("only", -3.0, (7.0, 7.0))])
    assert a.num_elites == 1
    assert a.sample_elites(5)[0].gene_id == "only"


def test_zero_successful_initial_individuals_fails_loudly():
    a = SlidingBoundariesArchive(dims=(4, 4), seed=0)
    with pytest.raises(ValueError, match="zero succeeded"):
        a.initialize([])
