"""Brax native environment adapter for PPO training.

Supports standard Brax environments (Ant, Humanoid, Walker2d, Halfcheetah, etc.)
using brax.envs registry and brax.training.agents.ppo for training.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from limen.adapters.base import EnvAdapter

logger = logging.getLogger(__name__)


class BraxAdapter(EnvAdapter):
    """Adapter for native Brax environments using Brax PPO.

    Works with any environment registered in brax.envs (Ant, Humanoid,
    Walker2d, Halfcheetah, Hopper, etc.). No hardcoded registry needed.
    """

    def __init__(self, config):
        self.config = config
        self._env_id = config.environment.env_id
        try:
            self._env = self._create_env()
        except Exception as e:
            import brax.envs
            available = list(brax.envs._envs.keys()) if hasattr(brax.envs, '_envs') else []
            raise ValueError(
                f"Cannot create Brax env {self._env_id!r}: {e}. "
                f"Available: {available}"
            ) from e

    def _create_env(self):
        """Create a fresh Brax environment instance."""
        import brax.envs
        return brax.envs.get_environment(self._env_id)

    def make_task_env(self):
        """Create a fresh task environment instance (public API for training)."""
        return self._create_env()

    def get_dummy_state(self) -> Any:
        rng = jax.random.key(0)
        state = jax.jit(self._env.reset)(rng)
        return state

    def get_dummy_action(self) -> jax.Array:
        return jnp.zeros(self._env.action_size)

    def is_continuous_action(self) -> bool:
        return True

    def fitness_type(self) -> str:
        """Locomotion tasks use unbounded fitness (forward displacement, like Eureka)."""
        return "reward"

    def fitness_key(self) -> str:
        """Ground-truth task fitness (forward displacement) is stored under success_rate."""
        return "success_rate"

    def get_default_obs_fn(self) -> Callable:
        """Return the default observation function matching the native env's _get_obs.

        For Brax envs, the native obs is qpos[2:] + qvel (Ant), etc.
        We just return state.obs which is already computed by the env.
        """
        def default_obs(state):
            obs = state.obs
            if isinstance(obs, dict):
                return obs["state"]
            return obs
        return default_obs

    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return None to use the env's built-in reward."""
        return None

    # ------------------------------------------------------------------
    # Brax training integration
    # ------------------------------------------------------------------

    def get_wrap_env_fn(self):
        """Native Brax environments use brax.envs.wrappers.training.wrap."""
        from brax.envs.wrappers import training as brax_training
        return brax_training.wrap

    def get_extract_success_fn(self):
        """Extract ground-truth task fitness from eval metrics.

        Following Eureka (ICLR 2024), the fitness function F is separate from
        the evolved reward. For locomotion envs, F = forward displacement
        (cur_dist - prev_dist summed over the episode).

        Brax envs report this under different metric keys:
        - Ant: forward_reward, reward_forward
        - Humanoid: forward_reward, reward_linvel
        - HalfCheetah: reward_run
        - Hopper/Walker2d: reward_forward
        - Swimmer: forward_reward, reward_fwd

        The evolved reward trains the policy, but this ground-truth metric
        ranks candidates in the evolutionary search.
        """
        # Ordered by preference — first match wins
        _FORWARD_KEYS = [
            "eval/episode_forward_reward",
            "eval/episode_reward_forward",
            "eval/episode_reward_run",
            "eval/episode_reward_linvel",
            "eval/episode_reward_fwd",
        ]

        def extract(metrics, episode_length):
            for key in _FORWARD_KEYS:
                val = metrics.get(key)
                if val is not None:
                    return float(val)
            # Fallback: use total evolved reward if no ground-truth metric found
            return float(metrics.get("eval/episode_reward", 0.0))
        return extract

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        from limen.train_brax import run_training
        return run_training(config, self, interface, obs_dim, total_timesteps)

    def run_training_multi_seed(self, config, interface, obs_dim, total_timesteps=None, num_seeds=None) -> Dict[str, Any]:
        from limen.train_brax import run_training_multi_seed
        return run_training_multi_seed(config, self, interface, obs_dim, total_timesteps, num_seeds)
