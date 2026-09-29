"""Elite replacement is by objective only; ties keep the incumbent."""

import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive


@pytest.fixture
def archive(rec):
    a = SlidingBoundariesArchive(dims=(2, 2), remap_frequency=0, seed=0)
    # edges dim0 [0,2,3], dim1 [0,2,3]
    a.initialize([rec("i0", 1.0, (0.0, 0.0)), rec("i1", 1.0, (1.0, 1.0)),
                  rec("i2", 1.0, (2.0, 2.0)), rec("i3", 1.0, (3.0, 3.0))])
    return a


def test_empty_cell_inserts(archive, rec):
    assert (0, 1) not in archive.elites
    res = archive.add(rec("new", -5.0, (0.5, 2.5)))
    assert res.cell == (0, 1) and res.inserted and res.replaced_gene_id is None
    assert archive.elites[(0, 1)].gene_id == "new"


def test_worse_candidate_rejected(archive, rec):
    incumbent = archive.elites[(0, 0)].gene_id
    res = archive.add(rec("worse", 0.5, (0.1, 0.1)))
    assert not res.inserted and archive.elites[(0, 0)].gene_id == incumbent


def test_better_candidate_replaces(archive, rec):
    incumbent = archive.elites[(0, 0)].gene_id
    res = archive.add(rec("better", 7.0, (0.1, 0.1)))
    assert res.inserted and res.replaced_gene_id == incumbent
    assert archive.elites[(0, 0)].gene_id == "better"


def test_equal_candidate_does_not_replace(archive, rec):
    incumbent = archive.elites[(0, 0)]
    res = archive.add(rec("tie", incumbent.objective, (0.1, 0.1)))
    assert not res.inserted and archive.elites[(0, 0)].gene_id == incumbent.gene_id


def test_initial_collision_keeps_best(rec):
    a = SlidingBoundariesArchive(dims=(1, 1), seed=0)
    a.initialize([rec("a", 1.0, (0, 0)), rec("b", 3.0, (1, 1)), rec("c", 3.0, (2, 2))])
    assert a.elites[(0, 0)].gene_id == "b"  # c ties b -> earlier record kept


def test_param_count_is_not_a_tiebreaker(rec):
    from sota.MujocoRL.mesb_archive import ArchiveRecord
    a = SlidingBoundariesArchive(dims=(1,), seed=0)
    a.initialize([ArchiveRecord("big", 0, 5.0, (0.0,), param_count=10**9)])
    a.add(ArchiveRecord("small", 1, 5.0, (0.0,), param_count=1))
    assert a.elites[(0,)].gene_id == "big"


def test_fixture_layout_and_coverage(archive):
    # edges are [0, 2, 3] on both axes: i0,i1 -> (0,0) ; i2,i3 -> (1,1); ties keep the first
    assert archive.boundaries == [[0.0, 2.0, 3.0], [0.0, 2.0, 3.0]]
    assert {c: r.gene_id for c, r in archive.elites.items()} == {(0, 0): "i0", (1, 1): "i2"}
    assert archive.num_elites == 2
    assert archive.coverage() == pytest.approx(2 / 4)
    assert archive.occupied_cells() == [(0, 0), (1, 1)]


def test_qd_score_raw_and_shifted(rec):
    a = SlidingBoundariesArchive(dims=(3,), seed=0, qd_score_offset=-100.0)
    a.initialize([rec("a", -10.0, (0.0,)), rec("b", 20.0, (1.0,)), rec("c", -5.0, (2.0,))])
    assert a.qd_score() == pytest.approx(5.0)            # negatives untouched
    assert a.shifted_qd_score() == pytest.approx(305.0)  # sum(obj + 100)
    assert a.mean_elite_objective() == pytest.approx(5.0 / 3)
    assert a.best_elite().gene_id == "b"
    b = SlidingBoundariesArchive(dims=(3,), seed=0)
    b.initialize([rec("a", -1.0, (0.0,))])
    assert b.shifted_qd_score() is None
