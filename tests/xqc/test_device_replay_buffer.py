"""DeviceReplayBuffer returns the same batches as XQC's numpy ParallelReplayBuffer.

Runs in the XQC environment (JAX + gymnasium):
    python -m pytest tests/xqc/test_device_replay_buffer.py
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("jax")
gym = pytest.importorskip("gymnasium")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sota", "XQC"))
from xqc import checkpoint  # noqa: E402
from xqc.replay_buffer import DeviceReplayBuffer, ParallelReplayBuffer  # noqa: E402

SEEDS, OBS, ACT, CAPACITY = 3, 5, 2, 50
FIELDS = ("observations", "actions", "rewards", "masks", "next_observations", "discount")


def make_pair(n_steps, gamma=0.99):
    obs_space = gym.spaces.Box(-1, 1, (SEEDS, OBS), np.float32)
    act_space = gym.spaces.Box(-1, 1, (SEEDS, ACT), np.float32)
    return tuple(cls(obs_space, act_space, CAPACITY, SEEDS, n_steps, gamma)
                 for cls in (ParallelReplayBuffer, DeviceReplayBuffer))


def fill(buffers, n, rng):
    for _ in range(n):
        transition = (rng.normal(size=(SEEDS, OBS)).astype(np.float32),
                      rng.uniform(-1, 1, (SEEDS, ACT)).astype(np.float32),
                      rng.normal(size=SEEDS).astype(np.float32),
                      (rng.random(SEEDS) > 0.1).astype(np.float32),   # mask 0 = terminal
                      rng.random(SEEDS) < 0.05,                       # truncation
                      rng.normal(size=(SEEDS, OBS)).astype(np.float32))
        for b in buffers:
            b.insert(*transition)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("inserted", [20, 50, 73])   # partly full, exactly full, wrapped
def test_same_batch_for_same_indices(n_steps, inserted):
    cpu, dev = make_pair(n_steps)
    fill((cpu, dev), inserted, np.random.default_rng(inserted))
    assert (cpu.size, cpu.insert_index) == (dev.size, dev.insert_index)

    np.random.seed(5)
    indxs = np.random.randint(cpu.size, size=(4, 16))
    np.random.seed(5)
    expected = cpu.sample_parallel_multibatch(16, 4)
    got = dev.gather(indxs)
    for field in FIELDS:
        e, g = np.asarray(getattr(expected, field)), np.asarray(getattr(got, field))
        assert e.shape == g.shape, field
        np.testing.assert_allclose(g, e, rtol=1e-6, atol=1e-6, err_msg=field)


def test_random_sample_only_reads_inserted_rows():
    _, dev = make_pair(1)
    obs_space_rows = 7
    for t in range(obs_space_rows):
        row = np.full((SEEDS, OBS), t, np.float32)
        dev.insert(row, np.zeros((SEEDS, ACT), np.float32), np.zeros(SEEDS, np.float32),
                   np.ones(SEEDS, np.float32), np.zeros(SEEDS, bool), row)
    for _ in range(5):
        batch = dev.sample_parallel_multibatch(32, 2)
        assert np.asarray(batch.observations).shape == (SEEDS, 2, 32, OBS)
        assert np.asarray(batch.observations).max() < obs_space_rows


def test_checkpoint_round_trip(tmp_path):
    cpu, dev = make_pair(1)
    fill((cpu, dev), 30, np.random.default_rng(0))
    checkpoint.save_buffer(str(tmp_path / "dev.npz"), dev)
    checkpoint.save_buffer(str(tmp_path / "cpu.npz"), cpu)
    _, restored = make_pair(1)
    checkpoint.load_buffer(str(tmp_path / "cpu.npz"), restored)   # a numpy checkpoint resumes on the GPU
    indxs = np.random.randint(30, size=(2, 8))
    for field in FIELDS:
        np.testing.assert_allclose(np.asarray(getattr(restored.gather(indxs), field)),
                                   np.asarray(getattr(dev.gather(indxs), field)), rtol=1e-6, atol=1e-6)
    with np.load(tmp_path / "dev.npz") as d:
        assert int(d["size"]) == 30 and d["observations"].shape == (SEEDS, 30, OBS)
