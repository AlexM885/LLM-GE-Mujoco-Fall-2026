"""TD3 training on Gymnasium MuJoCo environments for LIMEN.

Trains and scores one candidate interface on exactly the problem that
sota/MujocoRL/train_rl.py poses to LLM-GE:

- environment: eval.make_env (Walker2d-v5 with the pinned reward spec)
- learner: Stable-Baselines3 TD3 with the seed network.py (policy and TD3 kwargs)
- fitness: eval.evaluate_model, i.e. the mean undiscounted return of the
  environment's own reward over the evaluation episodes

The evolved get_observation replaces the observation the agent sees and the
evolved compute_reward replaces the reward it is trained on. The fitness is
always measured on the environment's own reward, never on the evolved one.

Each training runs in its own process (see adapters/gym_mujoco.py):
    python -m limen.train_gym --interface candidate.py --out metrics.json ...
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, NamedTuple, Optional

import gymnasium as gym
import jax
import numpy as np

# sota/MujocoRL holds eval.py and network.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval import evaluate_model, make_env, validate_reward_spec  # noqa: E402


class GymState(NamedTuple):
    """Simulator state handed to get_observation / compute_reward."""

    qpos: Any      # generalized positions, env.unwrapped.data.qpos
    qvel: Any      # generalized velocities, env.unwrapped.data.qvel
    obs: Any       # the environment's default observation
    step_num: Any  # steps taken in the current episode


def read_state(env, default_obs, step_num: int) -> GymState:
    """Build the state of a Gymnasium MuJoCo env as float32 numpy arrays."""
    data = env.unwrapped.data
    return GymState(
        qpos=np.asarray(data.qpos, dtype=np.float32),
        qvel=np.asarray(data.qvel, dtype=np.float32),
        obs=np.asarray(default_obs, dtype=np.float32),
        step_num=np.int32(step_num),
    )


def default_observation(state: GymState):
    """Baseline observation: the environment's own."""
    return state.obs


class InterfaceEnv(gym.Wrapper):
    """Replace observation and reward of a Gymnasium env with evolved functions.

    With use_evolved_reward=False the environment's own reward is returned, so
    the same wrapper serves for evaluating the task fitness.
    """

    def __init__(
        self,
        env,
        get_obs_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        obs_dim: Optional[int] = None,
    ):
        super().__init__(env)
        self._get_obs = jax.jit(get_obs_fn) if get_obs_fn is not None else None
        self._reward = jax.jit(reward_fn) if reward_fn is not None else None
        self.use_evolved_reward = True
        if self._get_obs is not None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        self._step_num = 0
        self._state = None
        # Evolved-reward return of every finished training episode
        self.episode_returns = []
        self._episode_return = 0.0
        # Length of every finished evaluation episode
        self.eval_lengths = []

    def _observe(self, state: GymState, default_obs):
        if self._get_obs is None:
            return default_obs
        obs = np.asarray(self._get_obs(state), dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("get_observation returned NaN or Inf")
        return obs

    def reset(self, **kwargs):
        default_obs, info = self.env.reset(**kwargs)
        self._step_num = 0
        self._episode_return = 0.0
        self._state = read_state(self.env, default_obs, self._step_num)
        return self._observe(self._state, default_obs), info

    def step(self, action):
        default_obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_num += 1
        prev_state = self._state
        self._state = read_state(self.env, default_obs, self._step_num)

        if self._reward is not None and self.use_evolved_reward:
            info["task_reward"] = reward
            reward = float(
                self._reward(prev_state, np.asarray(action, dtype=np.float32), self._state)
            )
            if not np.isfinite(reward):
                raise FloatingPointError("compute_reward returned NaN or Inf")

        if self.use_evolved_reward:
            self._episode_return += float(reward)
            if terminated or truncated:
                self.episode_returns.append(self._episode_return)
        elif terminated or truncated:
            self.eval_lengths.append(self._step_num)

        return self._observe(self._state, default_obs), reward, terminated, truncated, info


def _curve(values, points: int = 10):
    """Average a list into at most `points` consecutive chunks."""
    if len(values) < points:
        return [float(v) for v in values]
    chunks = np.array_split(np.asarray(values, dtype=float), points)
    return [float(c.mean()) for c in chunks]


def run_td3(
    get_obs_fn: Optional[Callable],
    reward_fn: Optional[Callable],
    obs_dim: Optional[int],
    env_id: str,
    total_timesteps: int,
    seed: int,
    eval_episodes: int,
    eval_max_steps: int,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train TD3 with the given interface and return the LIMEN metrics dict."""
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    import network  # the LLM-GE seed gene

    t0 = time.time()
    base_env = make_env(env_id)
    problems = validate_reward_spec(base_env)
    if problems:
        raise RuntimeError("fitness spec mismatch: " + "; ".join(problems))
    env = InterfaceEnv(base_env, get_obs_fn, reward_fn, obs_dim)

    # Same construction as train_rl.py
    policy_kwargs = network.get_policy_kwargs()
    td3_kwargs = network.get_td3_kwargs()
    policy_class = policy_kwargs.pop("policy_class", "MlpPolicy")
    exploration_noise = float(td3_kwargs.pop("exploration_noise", 0.1))
    if exploration_noise > 0 and "action_noise" not in td3_kwargs:
        n_actions = env.action_space.shape[-1]
        td3_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=exploration_noise * np.ones(n_actions)
        )
    if td3_kwargs.get("learning_starts", 100) >= total_timesteps:
        td3_kwargs["learning_starts"] = max(total_timesteps // 2, 1)

    model = TD3(
        policy=policy_class,
        env=env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        **td3_kwargs,
    )
    param_count = sum(p.numel() for p in model.actor.parameters())
    model.learn(total_timesteps=total_timesteps)
    if model_path is not None:
        model.save(model_path)

    # Task fitness: the environment's own reward, with the evolved observation
    env.use_evolved_reward = False
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        mean_reward, std_reward, _, eval_metrics = evaluate_model(
            model, env, num_episodes=eval_episodes, max_steps=eval_max_steps
        )
    learning_curve = _curve(env.episode_returns)
    final_length = float(np.mean(env.eval_lengths)) if env.eval_lengths else float(eval_max_steps)
    env.close()

    return {
        "final_return": float(mean_reward),
        "success_rate": float(mean_reward),
        "final_length": final_length,
        "std_reward": float(std_reward),
        "param_count": float(param_count),
        "mean_distance": eval_metrics["mean_distance"],
        "mean_control_cost": eval_metrics["mean_control_cost"],
        "foot_contact_0": eval_metrics.get("foot_contact_0", 0.0),
        "foot_contact_1": eval_metrics.get("foot_contact_1", 0.0),
        "mean_return_forward": eval_metrics["mean_return_forward"],
        "mean_return_healthy": eval_metrics["mean_return_healthy"],
        "mean_return_ctrl": eval_metrics["mean_return_ctrl"],
        "learning_curve": learning_curve,
        "success_curve": [],
        "training_time": time.time() - t0,
        "nan_detected": False,
        "model_path": model_path or "",
    }


def main():
    parser = argparse.ArgumentParser(description="Train TD3 with a LIMEN interface")
    parser.add_argument("--interface", type=str, required=True, help="Python file with the interface")
    parser.add_argument("--out", type=str, required=True, help="Where to write the metrics json")
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default=None, help="Save the trained model and its interface here")
    parser.add_argument("--default-obs", action="store_true", help="Keep the environment's observation")
    parser.add_argument("--default-reward", action="store_true", help="Keep the environment's reward")
    args = parser.parse_args()

    try:
        from limen.interface import MDPInterface

        required = []
        if not args.default_obs:
            required.append("get_observation")
        if not args.default_reward:
            required.append("compute_reward")

        get_obs_fn = reward_fn = obs_dim = None
        if required:
            interface = MDPInterface.from_file(args.interface, required_functions=required)
            if not args.default_obs:
                get_obs_fn = interface.get_observation
                probe = make_env(args.env_id)
                default_obs, _ = probe.reset(seed=0)
                obs_dim = interface.detect_obs_dim(read_state(probe, default_obs, 0))
                probe.close()
            if not args.default_reward:
                reward_fn = interface.compute_reward

        model_path = None
        if args.model_dir:
            # One model per candidate, named after its interface code so the two stay together
            os.makedirs(args.model_dir, exist_ok=True)
            with open(args.interface, "rb") as f:
                code = f.read()
            name = f"{hashlib.sha1(code).hexdigest()[:12]}_seed{args.seed}"
            with open(os.path.join(args.model_dir, name + ".py"), "wb") as f:
                f.write(code)
            model_path = os.path.join(args.model_dir, name + ".zip")

        result = run_td3(
            get_obs_fn, reward_fn, obs_dim, args.env_id, args.timesteps,
            args.seed, args.eval_episodes, args.eval_max_steps, model_path,
        )
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
