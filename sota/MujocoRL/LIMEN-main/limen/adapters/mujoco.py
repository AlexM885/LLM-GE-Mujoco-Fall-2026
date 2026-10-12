"""MuJoCo environment adapter for Brax PPO training.

Supports PandaTracking and Go1PushRecovery tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from limen.adapters.base import EnvAdapter
from limen.adapters.brax_wrapper import BraxMDPInterfaceWrapper

logger = logging.getLogger(__name__)

# Map env_id strings to task classes (lazy imports to avoid heavy deps at import time)
_TASK_REGISTRY = {
    "PandaPickAndTrack": "limen.tasks.panda_pick_and_track.PandaPickAndTrack",
    "Go1PushRecovery": "limen.tasks.go1_push_recovery.Go1PushRecovery",
}


def _import_class(dotpath: str):
    """Import a class from a dotted path string."""
    module_path, class_name = dotpath.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class MujocoAdapter(EnvAdapter):
    """Adapter for MuJoCo Playground environments using Brax PPO."""

    def __init__(self, config):
        self.config = config
        self._env_id = config.environment.env_id
        if self._env_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown MuJoCo env_id: {self._env_id!r}. "
                f"Available: {list(_TASK_REGISTRY.keys())}"
            )
        self._task_cls = _import_class(_TASK_REGISTRY[self._env_id])
        self._env = self._task_cls()

    def make_task_env(self):
        """Create a fresh task environment instance (public API for train_brax)."""
        return self._task_cls()

    def get_dummy_state(self) -> Any:
        """Get a real environment state for dry-run validation."""
        rng = jax.random.key(0)
        state = jax.jit(self._env.reset)(rng)
        return state

    def get_dummy_action(self) -> jax.Array:
        """Return a zero continuous action vector."""
        return jnp.zeros(self._env.action_size)

    def is_continuous_action(self) -> bool:
        return True

    def get_default_obs_fn(self) -> Callable:
        """Return a raw/sparse observation function — only what MuJoCo provides.

        No human-engineered features (error vectors, future predictions, etc).
        Just raw sensor data + bare minimum to know the task goal.
        This is the fair "starting point" obs for reward_only ablation.
        """
        if self._env_id == "PandaPickAndTrack":
            # 20-dim: qpos(7) + qvel(7) + gripper_pos(3) + target_pos(3)
            # Drops: error vectors, future predictions, target_vel, prev_ctrl
            env = self._env
            gripper_site = env._gripper_site
            arm_qposadr = env._robot_arm_qposadr

            def sparse_obs(state):
                data = state.data
                return jnp.concatenate([
                    data.qpos[arm_qposadr],                    # 7: arm joint positions
                    data.qvel[arm_qposadr],                    # 7: arm joint velocities
                    data.site_xpos[gripper_site],              # 3: gripper position
                    state.info["target_pos"],                  # 3: current target
                ])
            return sparse_obs

        elif self._env_id == "Go1PushRecovery":
            # Raw Go1 obs: qpos + qvel + basic body state
            # The Go1 default obs is already fairly raw (gyro, gravity, joints)
            # so just return the env's built-in obs
            def default_obs(state):
                return state.obs["state"]
            return default_obs

        # Fallback: use env's built-in obs
        env = self._env
        rng = jax.random.key(0)
        dummy_state = jax.eval_shape(env.reset, rng)
        obs_is_dict = isinstance(dummy_state.obs, dict)

        if obs_is_dict:
            def default_obs(state):
                return state.obs["state"]
        else:
            def default_obs(state):
                return state.obs

        return default_obs

    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return a sparse binary reward function (task completion, no shaping).

        This is the minimal reward a human would write — 1 if success, 0 otherwise.
        Used as the fixed reward in obs_only ablation mode.
        """
        if self._env_id == "PandaPickAndTrack":
            def sparse_reward(prev_state, action, next_state):
                mean_tracking = next_state.info["cumulative_tracking_error"] / jnp.maximum(
                    next_state.info["tracking_steps"], 1.0
                )
                return (mean_tracking < 0.02).astype(jnp.float32)
            return sparse_reward

        elif self._env_id == "Go1PushRecovery":
            def sparse_reward(prev_state, action, next_state):
                step_count = next_state.info["step_count"]
                cum_pos_error = next_state.info["cum_pos_error"]
                avg_pos_error = cum_pos_error / jnp.maximum(step_count.astype(float), 1.0)
                done = next_state.done.astype(bool)
                success = (~done) & (avg_pos_error < 0.10)
                return success.astype(jnp.float32)
            return sparse_reward

        return None

    # ------------------------------------------------------------------
    # Brax training integration
    # ------------------------------------------------------------------

    def get_wrap_env_fn(self):
        """MuJoCo Playground uses its own wrapper for Brax training."""
        from mujoco_playground._src import wrapper
        return wrapper.wrap_for_brax_training

    def get_extract_success_fn(self):
        """Extract success rate from MuJoCo eval metrics (eval/episode_success)."""
        def extract(metrics, episode_length):
            raw = float(metrics.get("eval/episode_success", 0.0))
            return raw / max(episode_length, 1)
        return extract

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        from limen.train_brax import run_training
        return run_training(config, self, interface, obs_dim, total_timesteps)

    def run_training_multi_seed(self, config, interface, obs_dim, total_timesteps=None, num_seeds=None) -> Dict[str, Any]:
        from limen.train_brax import run_training_multi_seed
        return run_training_multi_seed(config, self, interface, obs_dim, total_timesteps, num_seeds)
