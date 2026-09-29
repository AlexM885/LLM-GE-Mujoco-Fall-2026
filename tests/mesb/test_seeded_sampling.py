"""Parent sampling is uniform over occupied cells and reproducible."""

import random
from collections import Counter

import numpy as np
import pytest

from sota.MujocoRL.mesb_archive import SlidingBoundariesArchive


def _archive(rec, seed):
    a = SlidingBoundariesArchive(dims=(4, 4), seed=seed)
    a.initialize([rec(f"g{i}", float(i), (float(i % 4), float(i // 4))) for i in range(16)])
    assert a.num_elites == 16
    return a


def test_same_seed_same_parents(rec):
    s1 = [r.gene_id for r in _archive(rec, 7).sample_elites(50)]
    s2 = [r.gene_id for r in _archive(rec, 7).sample_elites(50)]
    s3 = [r.gene_id for r in _archive(rec, 8).sample_elites(50)]
    assert s1 == s2 and s1 != s3


def test_samples_only_occupied_elites(rec):
    a = _archive(rec, 0)
    elite_ids = {r.gene_id for r in a.elites.values()}
    assert {r.gene_id for r in a.sample_elites(200)} <= elite_ids


def test_uniform_over_cells(rec):
    a = _archive(rec, 0)
    counts = Counter(r.gene_id for r in a.sample_elites(16000))
    assert len(counts) == 16
    assert max(counts.values()) < 1200 and min(counts.values()) > 800


def test_without_replacement(rec):
    a = _archive(rec, 0)
    s = a.sample_elites(16, replace=False)
    assert len({r.gene_id for r in s}) == 16
    with pytest.raises(ValueError):
        a.sample_elites(17, replace=False)


def test_archive_does_not_touch_global_rngs(rec):
    random.seed(1)
    np.random.seed(1)
    py_state, np_state = random.getstate(), np.random.get_state()
    a = _archive(rec, 0)
    a.sample_elites(10)
    a.add_many([])
    assert random.getstate() == py_state
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(np.random.get_state(), np_state))


def test_empty_archive_sampling_raises():
    a = SlidingBoundariesArchive(dims=(2,))
    with pytest.raises(RuntimeError):
        a.sample_elites(1)
