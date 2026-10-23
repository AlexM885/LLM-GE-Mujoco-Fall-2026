"""state_dict -> JSON -> from_state_dict must restore the archive exactly."""

import json

from sota.MujocoRL.mesb_archive import (FixedGridArchive, SlidingBoundariesArchive,
                                        archive_from_state_dict)


def _populated(rec, **kw):
    a = SlidingBoundariesArchive(dims=(4, 3), remap_frequency=17, seed=123, **kw)
    a.initialize([rec(f"i{i}", float(i) * 0.3, (i * 1.1, -i * 0.7)) for i in range(9)])
    a.add_many([rec(f"g{i}", (i * 37 % 11) - 5.5, ((i * 13) % 29 / 3.0, (i * 7) % 17 * 0.25),
                    generation=1 + i // 10) for i in range(60)])
    a.sample_elites(7)  # advance the RNG so its state is non-trivial
    return a


def test_round_trip_identical(rec):
    a = _populated(rec, buffer_capacity=40, qd_score_offset=-10.0)
    state = json.loads(json.dumps(a.state_dict()))  # must be JSON-serialisable
    b = SlidingBoundariesArchive.from_state_dict(state)
    assert b.boundaries == a.boundaries
    assert b.buffer == a.buffer
    assert b.elites == a.elites
    assert b.total_seen == a.total_seen
    assert b.evaluations_since_remap == a.evaluations_since_remap
    assert b.last_remap_at == a.last_remap_at
    assert b.num_remaps == a.num_remaps and b.num_evicted == a.num_evicted
    assert b.remap_events == a.remap_events
    assert b.dims == a.dims and b.buffer_capacity == 40 and b.remap_frequency == 17
    # RNG state restored -> identical future samples
    assert [r.gene_id for r in b.sample_elites(20)] == [r.gene_id for r in a.sample_elites(20)]


def test_restored_archive_continues_identically(rec):
    a = _populated(rec)
    b = SlidingBoundariesArchive.from_state_dict(json.loads(json.dumps(a.state_dict())))
    more = [rec(f"z{i}", float(i % 5), (i * 0.9, i * -0.2), generation=9) for i in range(40)]
    ra, rb = a.add_many(more), b.add_many(more)
    assert ra == rb and a.state_dict() == b.state_dict()


def test_fixed_grid_round_trip(rec):
    a = FixedGridArchive(dims=(5, 5), ranges=[(0, 10), (-1, 1)], seed=1)
    a.initialize([rec("a", 1.0, (3.0, 0.2)), rec("b", 2.0, (9.0, -0.9))])
    b = archive_from_state_dict(json.loads(json.dumps(a.state_dict())))
    assert isinstance(b, FixedGridArchive) and b.ranges == a.ranges
    assert b.elites == a.elites and b.boundaries == a.boundaries


def test_mode_mismatch_rejected(rec):
    a = _populated(rec)
    state = a.state_dict()
    try:
        FixedGridArchive.from_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("expected mismatch error")
