"""Abstract base class defining the environment adapter interface.

To add a new environment adapter:
1. Subclass EnvAdapter
2. Implement the 3 required methods: get_dummy_state, get_default_obs_fn, get_default_reward_fn
3. Implement the training integration methods for your training backend
4. Register the adapter via register_adapter() in adapters/__init__.py
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


class EnvAdapter(ABC):
    """Base class for environment adapters.

    Required methods (must override):
        get_dummy_state      — return a real env state for dry-run/crash-filter
        get_default_obs_fn   — baseline observation function
        get_default_reward_fn — baseline reward function (or None for env built-in)

    Training integration (override for your training backend):
        run_training          — single-seed RL training
        run_training_multi_seed — multi-seed RL training

    Optional overrides:
        fitness_type, fitness_key, get_dummy_action, is_continuous_action,
        make_task_env, get_wrap_env_fn, get_extract_success_fn,
        make_env, make_eval_env, compute_success, num_actions
    """

    # ------------------------------------------------------------------
    # Required methods — every adapter must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dummy_state(self) -> Any:
        """Get a real environment state for dry-run validation.

        Returns:
            A state object from resetting the environment.
        """

    @abstractmethod
    def get_default_obs_fn(self) -> Callable:
        """Return the default (no-design-effort) observation function.

        This is used as the baseline observation when only reward is evolved.
        """

    @abstractmethod
    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return the default reward function, or None to use env built-in.

        Returns None when the environment's built-in reward should be used
        (e.g., for obs_only ablation mode).
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def fitness_type(self) -> str:
        """Return the fitness metric type for this adapter.

        "success_rate" — 0-to-1 fraction (default, used by discrete/goal tasks).
        "reward" — unbounded episode return (used by locomotion tasks like Eureka).

        Controls how fitness is displayed in prompts, how cascade thresholds
        are interpreted, and which metric key is used for ranking.
        """
        return "success_rate"

    def fitness_key(self) -> str:
        """Return the metrics dict key used as fitness."""
        return "success_rate" if self.fitness_type() == "success_rate" else "final_return"

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        """Run RL training. Discrete adapters use train.py; MuJoCo uses train_brax.py."""
        from limen.train import run_training
        return run_training(config, self, interface, obs_dim, total_timesteps)

    def run_training_multi_seed(self, config, interface, obs_dim, total_timesteps=None, num_seeds=None) -> Dict[str, Any]:
        """Run RL training across multiple seeds.

        Default: loops over run_training() with different seeds.
        MuJoCo adapter overrides this to share compiled train_fn across seeds.
        """
        import dataclasses
        tc = config.training
        n = num_seeds or tc.num_seeds
        if n <= 1:
            return self.run_training(config, interface, obs_dim, total_timesteps)

        all_success, all_return, all_length, all_curves = [], [], [], []
        total_time = 0.0
        for i in range(n):
            seed = tc.seed + i
            mod_config = dataclasses.replace(
                config, training=dataclasses.replace(tc, seed=seed),
            )
            m = self.run_training(mod_config, interface, obs_dim, total_timesteps)
            all_success.append(m["success_rate"])
            all_return.append(m["final_return"])
            all_length.append(m["final_length"])
            total_time += m.get("training_time", 0.0)
            if m.get("success_curve"):
                all_curves.append(m["success_curve"])

        avg_curve = []
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            avg_curve = [sum(c[j] for c in all_curves) / n for j in range(min_len)]

        return {
            "final_return": sum(all_return) / n,
            "final_length": sum(all_length) / n,
            "success_rate": sum(all_success) / n,
            "success_curve": avg_curve,
            "learning_curve": [],
            "training_time": total_time,
        }

    def get_dummy_action(self) -> jax.Array:
        """Return a dummy action for crash filter validation."""
        return jnp.int32(0)

    def is_continuous_action(self) -> bool:
        """Whether the action space is continuous."""
        return False

    def get_wrap_env_fn(self) -> Any:
        """Return the env wrapping function for Brax PPO training.

        Used by train_brax.py. Examples:
        - brax.envs.wrappers.training.wrap (native Brax envs)
        - mujoco_playground._src.wrapper.wrap_for_brax_training (MuJoCo Playground)

        Only needed for Brax-based adapters. Returns None for non-Brax adapters.
        """
        return None

    def get_extract_success_fn(self) -> Any:
        """Return a function to extract success rate from Brax eval metrics.

        Signature: (eval_metrics: dict, episode_length: int) -> float

        If None, the training module uses reward as success (reward-based fitness).
        If provided, extracts a [0,1] success rate from the eval metrics dict.
        """
        return None

    def make_task_env(self) -> Any:
        """Create a fresh base environment instance for training.

        Used by Brax training to create wrapped/eval environments.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Discrete-env methods (used by train.py / xminigrid)
    # Not required for Brax-based adapters.
    # ------------------------------------------------------------------

    def make_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
    ) -> Tuple[Any, Any]:
        """Create a wrapped environment ready for training.

        Returns:
            (env, env_params) tuple.
        """
        raise NotImplementedError

    def make_eval_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
        max_steps: int,
    ) -> Tuple[Any, Any]:
        """Create a wrapped environment for evaluation with custom max_steps.

        Returns:
            (env, env_params) tuple with max_steps overridden.
        """
        raise NotImplementedError

    def compute_success(self, rollout_stats: Any, env_params: Any) -> jax.Array:
        """Compute per-episode success from rollout statistics.

        Returns:
            Float array of per-episode success (1.0 = success, 0.0 = failure).
        """
        raise NotImplementedError

    def num_actions(self, env_params: Any) -> int:
        """Return the number of discrete actions for this environment."""
        raise NotImplementedError
