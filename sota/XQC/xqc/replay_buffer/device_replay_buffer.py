"""ParallelReplayBuffer kept in accelerator memory.

Same interface and same batches as :class:`ParallelReplayBuffer`: the buffer
arrays live on the JAX default device (the GPU in a gene job), transitions are
written there one step at a time, and batches are sampled there, so a training
step no longer gathers a batch with numpy and copies it to the GPU.

The sampling arithmetic (n-step returns, the newest transition treated as a
truncation, discounts) is the numpy version's, line for line; only the random
indices come from ``jax.random`` instead of ``np.random``. Tests:
tests/xqc/test_device_replay_buffer.py.
"""

import functools

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from xqc.replay_buffer.parallel_replay_buffer import Batch

FIELDS = ("observations", "next_observations", "actions", "rewards", "masks", "timeouts")


@functools.partial(jax.jit, donate_argnums=(0,))
def _insert(arrays, index, observation, action, reward, mask, trunc, next_observation):
    values = (observation, next_observation, action, reward, mask, trunc)
    return tuple(a.at[:, index].set(jnp.asarray(v, a.dtype)) for a, v in zip(arrays, values))


@functools.partial(jax.jit, static_argnames=("n_steps", "gamma"))
def _gather(arrays, indxs, insert_index, n_steps, gamma):
    observations, next_observations, actions, rewards, masks, timeouts = arrays
    num_seeds, capacity = rewards.shape

    # The newest transition has no successor yet: treat it as a truncation.
    last_valid_idx = (insert_index - 1) % capacity
    timeouts = timeouts.at[:, last_valid_idx].set(
        jnp.logical_or(timeouts[:, last_valid_idx], masks[:, last_valid_idx]).astype(timeouts.dtype))

    steps = jnp.arange(n_steps).reshape(1, -1)
    indices = (indxs[..., None] + steps) % capacity

    rewards_seq = rewards[:, indices]
    done_seq = jnp.logical_not(masks[:, indices])
    truncated_seq = timeouts[:, indices]

    done_or_trunc = jnp.logical_or(done_seq, truncated_seq)
    done_idx = done_or_trunc.argmax(axis=-1)
    has_done_or_truncated = done_or_trunc.any(axis=-1)
    done_idx = jnp.where(has_done_or_truncated, done_idx, n_steps - 1)

    mask = jnp.arange(n_steps).reshape(1, 1, 1, -1) <= done_idx[..., None]
    target_q_discounts = gamma ** mask.sum(axis=-1, keepdims=True).astype(jnp.float32)

    discounts = gamma ** jnp.arange(n_steps, dtype=jnp.float32).reshape(1, 1, 1, -1)
    n_step_returns = (rewards_seq * discounts * mask).sum(axis=-1)

    last_indices = (indxs[None] + done_idx) % capacity
    seed_idx = jnp.arange(num_seeds)[:, None, None]
    return Batch(
        observations=observations[:, indxs],
        actions=actions[:, indxs],
        rewards=n_step_returns,
        masks=masks[seed_idx, last_indices],
        next_observations=next_observations[seed_idx, last_indices],
        discount=target_q_discounts,
    )


@functools.partial(jax.jit, static_argnames=("batch_size", "num_batches", "n_steps", "gamma"))
def _sample(arrays, key, size, insert_index, batch_size, num_batches, n_steps, gamma):
    key, sample_key = jax.random.split(key)
    indxs = jax.random.randint(sample_key, (num_batches, batch_size), 0, size)
    return key, _gather(arrays, indxs, insert_index, n_steps, gamma)


class DeviceReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        capacity: int,
        num_seeds: int,
        n_steps: int,
        gamma: float,
    ):
        obs_dim, act_dim = observation_space.shape[-1], action_space.shape[-1]
        # float32 throughout: the learner computes in float32 either way.
        self._arrays = (
            jnp.zeros((num_seeds, capacity, obs_dim), jnp.float32),
            jnp.zeros((num_seeds, capacity, obs_dim), jnp.float32),
            jnp.zeros((num_seeds, capacity, act_dim), jnp.float32),
            jnp.zeros((num_seeds, capacity), jnp.float32),
            jnp.zeros((num_seeds, capacity), jnp.float32),
            jnp.zeros((num_seeds, capacity), jnp.float32),
        )
        self.num_seeds = num_seeds
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = float(gamma)
        self.size = 0
        self.insert_index = 0
        # Seeded from np.random, which the trainers seed with cfg.seed.
        self._key = jax.random.PRNGKey(np.random.randint(2**31 - 1))

    def __getattr__(self, name):
        # checkpoint.save_buffer reads the arrays by field name.
        if name in FIELDS:
            return self.__dict__["_arrays"][FIELDS.index(name)]
        raise AttributeError(name)

    def insert(self, observation, action, reward, mask, truncs, next_observation):
        self._arrays = _insert(self._arrays, self.insert_index, observation, action, reward,
                               mask, truncs, next_observation)
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        self._key, batch = _sample(self._arrays, self._key, self.size, self.insert_index,
                                   batch_size, num_batches, self.n_steps, self.gamma)
        return batch

    def gather(self, indxs) -> Batch:
        """The batch for given indices (num_batches x batch_size); used by the tests."""
        return _gather(self._arrays, jnp.asarray(indxs), self.insert_index, self.n_steps, self.gamma)

    def load_arrays(self, arrays: dict, size: int, insert_index: int):
        """Restore from checkpoint.load_buffer's arrays (first `size` rows of each field)."""
        self._arrays = tuple(a.at[:, :size].set(jnp.asarray(arrays[f], a.dtype))
                             for f, a in zip(FIELDS, self._arrays))
        self.size = size
        self.insert_index = insert_index
