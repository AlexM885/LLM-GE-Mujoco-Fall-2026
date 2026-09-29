"""Deterministic cell indexing under empirical boundaries."""

import pytest

from sota.MujocoRL.mesb_archive import (ArchiveNotInitializedError, FixedGridArchive,
                                        InvalidRecordError, SlidingBoundariesArchive)


@pytest.fixture
def archive(rec):
    # dim0 edges [0, 2, 4, 6, 8, 9], dim1 edges [0, 5, 9]
    a = SlidingBoundariesArchive(dims=(5, 2), seed=0)
    a.initialize([rec(f"g{i}", 0.0, (float(i), float(i))) for i in range(10)])
    assert a.boundaries == [[0, 2, 4, 6, 8, 9], [0, 5, 9]]
    return a


def test_minimum_goes_to_first_cell(archive):
    assert archive.index_of((0.0, 0.0)) == (0, 0)


def test_maximum_goes_to_last_cell(archive):
    assert archive.index_of((9.0, 9.0)) == (4, 1)


def test_exact_internal_boundary_goes_to_upper_cell(archive):
    assert archive.index_of((2.0, 5.0)) == (1, 1)
    assert archive.index_of((1.999, 4.999)) == (0, 0)
    assert archive.index_of((8.0, 0.0)) == (4, 0)


def test_below_range_clips_to_edge_cell(archive):
    assert archive.index_of((-1e9, -3.0)) == (0, 0)


def test_above_range_clips_to_edge_cell(archive):
    assert archive.index_of((1e9, 42.0)) == (4, 1)


def test_duplicate_boundaries(rec):
    a = SlidingBoundariesArchive(dims=(4,), seed=0)
    a.initialize([rec("a", 0.0, (1.0,)), rec("b", 0.0, (1.0,)), rec("c", 0.0, (1.0,)),
                  rec("d", 0.0, (2.0,))])
    # N=4,B=4 ranks 1,2,3 -> edges [1, 1, 1, 2, 2]
    assert a.boundaries == [[1.0, 1.0, 1.0, 2.0, 2.0]]
    assert a.index_of((0.5,)) == (0,)
    assert a.index_of((1.0,)) == (2,)   # side='right' past both duplicate 1.0 edges
    assert a.index_of((1.5,)) == (2,)
    assert a.index_of((2.0,)) == (3,)
    assert a.degenerate_dimensions() == [0]


def test_index_is_deterministic(archive):
    assert all(archive.index_of((3.3, 7.7)) == (1, 1) for _ in range(100))


def test_invalid_measures_rejected(archive):
    with pytest.raises(InvalidRecordError):
        archive.index_of((float("nan"), 1.0))
    with pytest.raises(InvalidRecordError):
        archive.index_of((1.0,))


def test_requires_initialisation():
    with pytest.raises(ArchiveNotInitializedError):
        SlidingBoundariesArchive(dims=(2, 2)).index_of((0.0, 0.0))


def test_fixed_grid_matches_legacy_get_bin(rec):
    # Legacy rule from origin/Mujoco-ME-Base@6b73ae167: min(int(frac * B), B - 1)
    ranges = [(-500.0, 2000.0), (0.0, 10.0)]
    a = FixedGridArchive(dims=(20, 20), ranges=ranges)
    a.initialize([rec("x", 1.0, (0.0, 0.0))])
    for x, y in [(-600, -1), (-500, 0), (0, 0.49), (1999.99, 9.99), (2000, 10), (5000, 50), (125, 0.5)]:
        expect = tuple(min(int(min(max((v - lo) / (hi - lo), 0), 1) * 20), 19)
                       for v, (lo, hi) in zip((x, y), ranges))
        assert a.index_of((x, y)) == expect
