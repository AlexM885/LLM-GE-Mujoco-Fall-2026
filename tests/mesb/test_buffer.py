"""Buffer (xi) semantics."""

import math

import pytest

from sota.MujocoRL.mesb_archive import InvalidRecordError, SlidingBoundariesArchive


def test_unlimited_buffer_keeps_every_successful_record(rec):
    a = SlidingBoundariesArchive(dims=(3, 3), remap_frequency=7, buffer_capacity=None, seed=0)
    a.initialize([rec(f"i{i}", 0.0, (i, i)) for i in range(5)])
    a.add_many([rec(f"g{i}", float(i), (i % 13, i % 5)) for i in range(200)])
    assert len(a.buffer) == a.total_seen == 205
    assert [r.gene_id for r in a.buffer][:5] == [f"i{i}" for i in range(5)]
    assert a.num_evicted == 0


def test_finite_buffer_drops_oldest(rec):
    a = SlidingBoundariesArchive(dims=(2,), remap_frequency=0, buffer_capacity=3, seed=0)
    a.initialize([rec("a", 0.0, (0,)), rec("b", 0.0, (1,))])
    a.add_many([rec("c", 0.0, (2,)), rec("d", 0.0, (3,)), rec("e", 0.0, (4,))])
    assert [r.gene_id for r in a.buffer] == ["c", "d", "e"]
    assert a.total_seen == 5 and a.num_evicted == 2


def test_finite_buffer_remap_uses_only_buffer(rec):
    a = SlidingBoundariesArchive(dims=(2,), remap_frequency=0, buffer_capacity=2, seed=0)
    a.initialize([rec("old", 100.0, (0.0,)), rec("o2", 1.0, (1.0,))])
    a.add_many([rec("n1", 1.0, (50.0,)), rec("n2", 1.0, (60.0,))])
    a.remap()
    assert a.boundaries == [[50.0, 60.0, 60.0]]
    # "old" was evicted from the buffer so it does not survive the remap (paper/pyribs).
    assert {r.gene_id for r in a.elites.values()} == {"n1", "n2"}


def test_failed_or_nonfinite_records_never_enter_buffer(rec):
    a = SlidingBoundariesArchive(dims=(2,), seed=0)
    a.initialize([rec("ok", 1.0, (0.0,))])
    for bad in (rec("nan_obj", math.nan, (0.0,)), rec("inf_meas", 1.0, (math.inf,)),
                rec("nan_meas", 1.0, (math.nan,)), rec("neg_inf", -math.inf, (0.0,))):
        with pytest.raises(InvalidRecordError):
            a.add(bad)
    with pytest.raises(InvalidRecordError):
        a.add(rec("wrong_dim", 1.0, (0.0, 1.0)))
    assert [r.gene_id for r in a.buffer] == ["ok"] and a.total_seen == 1


def test_invalid_capacity():
    with pytest.raises(ValueError):
        SlidingBoundariesArchive(dims=(2,), buffer_capacity=0)
