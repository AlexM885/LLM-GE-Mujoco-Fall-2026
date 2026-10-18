"""Shared MDP interface wrapper for Brax-based environments.

Intercepts state.obs and state.reward with evolved functions.
Used by both MuJoCo Playground (mujoco_adapter) and native Brax (brax_adapter).
"""

import jax.numpy as jnp


class BraxMDPInterfaceWrapper:
    """Wraps a Brax-style env, replacing obs and reward with evolved functions.

    Works with any environment that has:
    - reset(rng) -> State
    - step(state, action) -> State
    - State.replace(obs=..., reward=...)
    - observation_size, action_size properties

    The wrapper returns obs as {"state": array} for compatibility with
    Brax PPO's policy_obs_key="state" / value_obs_key="state".
    """

    def __init__(self, env, get_obs_fn=None, reward_fn=None, obs_dim=None):
        self._env = env
        self._get_obs_fn = get_obs_fn
        self._reward_fn = reward_fn
        self._obs_dim = obs_dim

    def reset(self, rng):
        state = self._env.reset(rng)
        if self._get_obs_fn is not None:
            new_obs = self._get_obs_fn(state)
            new_obs = jnp.asarray(new_obs)
            state = state.replace(obs={"state": new_obs})
        return state

    def step(self, state, action):
        prev_state = state
        next_state = self._env.step(state, action)
        if self._get_obs_fn is not None:
            new_obs = self._get_obs_fn(next_state)
            new_obs = jnp.asarray(new_obs)
            next_state = next_state.replace(obs={"state": new_obs})
        if self._reward_fn is not None:
            new_reward = self._reward_fn(prev_state, action, next_state)
            new_reward = jnp.asarray(new_reward).squeeze()
            next_state = next_state.replace(reward=new_reward)
        return next_state

    @property
    def observation_size(self):
        if self._get_obs_fn is not None and self._obs_dim is not None:
            return {"state": (self._obs_dim,)}
        return self._env.observation_size

    @property
    def action_size(self):
        return self._env.action_size

    def __getattr__(self, name):
        return getattr(self._env, name)
