"""Gymnasium MuJoCo environment adapter (TD3 training).

Runs LIMEN on the same problem as LLM-GE's sota/MujocoRL/train_rl.py:
Walker2d-v5 built by eval.make_env, trained with Stable-Baselines3 TD3 and
scored by eval.evaluate_model. See limen/train_gym.py.

Every training is launched as a separate process so that the candidates of a
batch (candidates_per_iteration) train in parallel.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from limen.adapters.base import EnvAdapter
from limen.train_gym import GymState, default_observation, make_env, read_state

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


class GymMujocoAdapter(EnvAdapter):
    """Adapter for Gymnasium MuJoCo environments trained with SB3 TD3."""

    def __init__(self, config):
        self.config = config
        self._env_id = config.environment.env_id
        env = make_env(self._env_id)
        default_obs, _ = env.reset(seed=0)
        state = read_state(env, default_obs, 0)
        self._dummy_state = GymState(*(jnp.asarray(x) for x in state))
        self._action_size = int(env.action_space.shape[0])
        env.close()

    def get_dummy_state(self) -> Any:
        return self._dummy_state

    def get_dummy_action(self) -> jax.Array:
        return jnp.zeros(self._action_size)

    def is_continuous_action(self) -> bool:
        return True

    def fitness_type(self) -> str:
        """Fitness is the unbounded return of the environment's own reward."""
        return "reward"

    def get_default_obs_fn(self) -> Callable:
        return default_observation

    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return None to use the env's built-in reward."""
        return None

    # ------------------------------------------------------------------
    # Training integration
    # ------------------------------------------------------------------

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        tc = config.training
        if total_timesteps is None:
            total_timesteps = tc.total_timesteps

        if tc.gym_work_dir:
            os.makedirs(tc.gym_work_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="limen_gym_", dir=tc.gym_work_dir) as tmp:
            code_path = os.path.join(tmp, "interface.py")
            out_path = os.path.join(tmp, "metrics.json")
            with open(code_path, "w") as f:
                f.write(interface.source_code or "")

            cmd = shlex.split(tc.gym_launcher) if tc.gym_launcher else []
            cmd += [
                sys.executable, "-m", "limen.train_gym",
                "--interface", code_path,
                "--out", out_path,
                "--env-id", self._env_id,
                "--timesteps", str(total_timesteps),
                "--seed", str(tc.seed),
                "--eval-episodes", str(tc.eval_episodes_for_fitness),
                "--eval-max-steps", str(tc.eval_max_steps),
            ]
            if tc.gym_model_dir:
                cmd += ["--model-dir", tc.gym_model_dir]
            if interface.get_observation is None or interface.get_observation is default_observation:
                cmd.append("--default-obs")
            if interface.compute_reward is None:
                cmd.append("--default-reward")

            # The interface functions are tiny; keep JAX off the GPU that TD3 uses
            env = dict(os.environ, JAX_PLATFORMS="cpu")
            logger.info(
                "Starting TD3 training: %s steps, env=%s, seed=%d",
                f"{total_timesteps:,}", self._env_id, tc.seed,
            )
            proc = subprocess.run(
                cmd, cwd=str(_PACKAGE_ROOT), env=env, capture_output=True, text=True,
                timeout=config.evaluator.timeout,
            )
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"training process produced no metrics (exit {proc.returncode}): "
                    f"{proc.stderr[-2000:]}"
                )
            with open(out_path) as f:
                metrics = json.load(f)

        if "error" in metrics:
            logger.debug("Training traceback:\n%s", metrics.get("traceback", ""))
            raise RuntimeError(metrics["error"])
        return metrics
