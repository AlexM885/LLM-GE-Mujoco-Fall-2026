"""Remapping moves boundaries, rebuilds cells, and keeps the best on collision."""

import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive


def _cells(a):
    return {c: r.gene_id for c, r in a.elites.items()}


def test_remap_moves_boundaries_and_reassigns_elites(rec):
    a = SlidingBoundariesArchive(dims=(2,), remap_frequency=4, seed=0)
    # Initial values 0,1,2,3 -> edges [0, 2, 3]
    a.initialize([rec("a", 1.0, (0.0,)), rec("b", 5.0, (1.0,)),
                  rec("c", 2.0, (2.0,)), rec("d", 9.0, (3.0,))])
    assert a.boundaries == [[0.0, 2.0, 3.0]]
    assert _cells(a) == {(0,): "b", (1,): "d"}

    # Four far-right points: before remap they all land in the top cell.
    results = a.add_many([rec(f"e{i}", 0.5, (10.0 + i,)) for i in range(4)])
    assert [r.cell for r in results] == [(1,)] * 4
    assert [r.remapped for r in results] == [False, False, False, True]
    assert a.num_remaps == 1 and a.evaluations_since_remap == 0 and a.last_remap_at == 8

    # N=8 sorted [0,1,2,3,10,11,12,13] -> rank 4 -> edge 10.0
    assert a.boundaries == [[0.0, 10.0, 13.0]]
    # Old top-cell elite "d" (3.0) now shares cell 0 with a,b,c and wins (9.0).
    # Cell 1 now holds the e* records; all tie at 0.5 -> the earliest, e0.
    assert _cells(a) == {(0,): "d", (1,): "e0"}


def test_collision_after_remap_keeps_highest_objective(rec):
    a = SlidingBoundariesArchive(dims=(3,), remap_frequency=3, seed=0)
    a.initialize([rec("low", 1.0, (0.0,)), rec("mid", 2.0, (5.0,)), rec("high", 3.0, (10.0,))])
    assert len(a.elites) == 3
    # Adding three far outliers compresses the old points into one cell.
    a.add_many([rec(f"far{i}", 0.0, (1000.0 + i,)) for i in range(3)])
    assert a.num_remaps == 1
    # N=6 sorted [0,5,10,1000,1001,1002]: ranks 2,4 -> edges [0,10,1001,1002]
    assert a.boundaries == [[0.0, 10.0, 1001.0, 1002.0]]
    assert _cells(a) == {(0,): "mid", (1,): "high", (2,): "far1"}
    # "far0" (1000) shares cell 1 with "high" and loses on objective.


def test_remap_rebuilds_even_with_same_boundaries(rec):
    a = SlidingBoundariesArchive(dims=(2,), remap_frequency=0, seed=0)
    a.initialize([rec("a", 1.0, (0.0,)), rec("b", 1.0, (1.0,))])
    before = _cells(a)
    a.remap()
    assert _cells(a) == before and a.num_remaps == 1


def test_generation_end_mode_defers_remap(rec):
    a = SlidingBoundariesArchive(dims=(2,), remap_frequency=2, seed=0,
                                 remap_at_generation_end=True)
    a.initialize([rec("a", 1.0, (0.0,)), rec("b", 1.0, (1.0,))])
    res = a.add_many([rec(f"x{i}", 1.0, (100.0 + i,)) for i in range(5)])
    assert not any(r.remapped for r in res) and a.num_remaps == 0
    assert a.boundaries == [[0.0, 1.0, 1.0]]  # unchanged mid-generation
    assert a.end_generation() is True
    # N=7 sorted [0,1,100,101,102,103,104] -> rank floor(7/2)=3 -> 101
    assert a.num_remaps == 1 and a.boundaries[0][1] == 101.0
    assert a.end_generation() is False  # nothing new since


def test_paper_mode_remaps_every_delta(rec):
    a = SlidingBoundariesArchive(dims=(4, 4), remap_frequency=100, seed=0)
    a.initialize([rec(f"i{i}", 0.0, (float(i), float(-i))) for i in range(10)])
    a.add_many([rec(f"g{i}", float(i), (float(i % 37), float(i % 11))) for i in range(350)])
    assert a.num_remaps == 3 and a.evaluations_since_remap == 50
    assert [e["total_seen"] for e in a.remap_events] == [10, 110, 210, 310]
    assert a.end_generation() is False  # paper mode never remaps at generation end


def test_remap_on_uninitialised_archive_raises():
    with pytest.raises(RuntimeError):
        SlidingBoundariesArchive(dims=(2,)).remap()
