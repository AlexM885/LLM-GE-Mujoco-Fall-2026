"""Boundaries must be empirical ranks of observed data, not uniform bins."""

import numpy as np
import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive, compute_boundaries_1d


def test_rank_rule_on_known_sorted_data():
    # N=10, B=5: internal ranks floor(j*10/5) = 2, 4, 6, 8
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = compute_boundaries_1d(values, 5)
    assert edges.tolist() == [0, 2, 4, 6, 8, 9]


def test_rank_rule_non_divisible():
    # N=7, B=3: ranks floor(7/3)=2, floor(14/3)=4 -> values 30, 50
    values = [70, 10, 40, 20, 60, 30, 50]  # unsorted on purpose
    edges = compute_boundaries_1d(values, 3)
    assert edges.tolist() == [10, 30, 50, 70]


def test_boundaries_are_observed_values_and_not_uniform():
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(size=500))  # heavily skewed
    edges = compute_boundaries_1d(values, 10)
    assert set(edges.tolist()) <= set(values.tolist())  # no interpolation
    uniform = np.linspace(values.min(), values.max(), 11)
    assert not np.allclose(edges, uniform)
    assert edges[0] == values.min() and edges[-1] == values.max()


def test_equal_mass_per_cell():
    values = np.arange(1000.0)
    edges = compute_boundaries_1d(values, 4)
    counts = np.histogram(values, bins=edges)[0]
    assert counts.tolist() == [250, 250, 250, 250]


def test_fewer_values_than_bins_gives_duplicate_edges():
    edges = compute_boundaries_1d([5.0, 7.0], 4)
    # N=2, B=4: ranks floor(2/4)=0, floor(4/4)=1, floor(6/4)=1
    assert edges.tolist() == [5.0, 5.0, 7.0, 7.0, 7.0]


def test_single_bin_and_single_value():
    assert compute_boundaries_1d([3.0, 1.0], 1).tolist() == [1.0, 3.0]
    assert compute_boundaries_1d([2.0], 3).tolist() == [2.0, 2.0, 2.0, 2.0]


def test_errors():
    with pytest.raises(ValueError):
        compute_boundaries_1d([], 3)
    with pytest.raises(ValueError):
        compute_boundaries_1d([1.0], 0)
    with pytest.raises(ValueError):
        compute_boundaries_1d([1.0, float("nan")], 2)


def test_archive_initialises_from_data_without_ranges(rec):
    a = SlidingBoundariesArchive(dims=(2, 3), seed=0)
    records = [rec(f"g{i}", float(i), (i * 10.0, 100.0 - i)) for i in range(6)]
    a.initialize(records)
    # dim0 values 0..50 (N=6,B=2): rank 3 -> 30 ; dim1 values 95..100 (B=3): ranks 2,4 -> 97, 99
    assert a.boundaries == [[0.0, 30.0, 50.0], [95.0, 97.0, 99.0, 100.0]]


def test_boundaries_follow_distribution_shift(rec):
    a = SlidingBoundariesArchive(dims=(4,), remap_frequency=50, seed=0)
    a.initialize([rec(f"a{i}", 0.0, (float(i),)) for i in range(20)])      # 0..19
    first = a.boundaries[0]
    a.add_many([rec(f"b{i}", 0.0, (100.0 + i,)) for i in range(50)])       # triggers remap
    second = a.boundaries[0]
    assert a.num_remaps == 1
    assert first != second
    assert second[-1] == 149.0 and second[0] == 0.0
