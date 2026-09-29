"""Standardised, deterministic post-training evaluation for MuJoCo genes.

Produces the MESB quality and behaviour descriptors:

* quality     ``mean_reward``        mean undiscounted episode return
* descriptor  ``mean_distance``      mean of (final_x - initial_x) per episode
* descriptor  ``mean_control_cost``  mean of the per-episode *total* positive
                                     control cost

Verified for HalfCheetah-v4 (gymnasium 1.2.3, mujoco 3.14):

* ``reset()`` returns an empty info dict, so the initial x position is read
  from ``env.unwrapped.data.qpos[0]`` (the root x slide joint) right after the
  seeded reset; after every step ``info["x_position"]`` is used, which equals
  ``qpos[0]``.
* ``step()`` reports ``info["reward_ctrl"] = -ctrl_cost_weight * sum(a**2)``
  with ``ctrl_cost_weight = 0.1``; the positive cost is ``-reward_ctrl``.
  If an environment does not report ``reward_ctrl`` we fall back to the
  environment's *own* ``control_cost(action)`` method (the same function its
  reward uses). If neither exists we fail loudly instead of guessing.

Every evaluation episode ``i`` is reset with ``seed = base_seed + i`` so all
genes are compared on identical start states, independent of the stochastic
training seed.

Standalone use (evaluate a saved model and optionally record a video)::

    python sota/MujocoRL/behavior_eval.py --model path/to/model.zip \
        --seed 1000 --episodes 10 --max-steps 1000 --video best.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_ENV_ID = os.getenv("MUJOCO_ENV_ID", "HalfCheetah-v4")

PolicyFn = Callable[[np.ndarray], np.ndarray]


class EvaluationError(RuntimeError):
    """Raised when an episode produces unusable (non-finite / unmeasurable) data."""


def make_env(env_id: str = DEFAULT_ENV_ID, render_mode: str | None = None):
    import gymnasium as gym
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return gym.make(env_id, **kwargs)


def _x_position(env, info: dict | None) -> tuple[float, str]:
    if isinstance(info, dict) and info.get("x_position") is not None:
        return float(info["x_position"]), "info.x_position"
    try:
        return float(env.unwrapped.data.qpos[0]), "qpos[0]"
    except (AttributeError, IndexError, TypeError) as exc:
        raise EvaluationError(f"cannot read x position from {env.spec.id}: {exc}") from exc


def _control_cost(env, info: dict, action: np.ndarray) -> tuple[float, str]:
    if "reward_ctrl" in info:
        return -float(info["reward_ctrl"]), "info.reward_ctrl"
    fn = getattr(env.unwrapped, "control_cost", None)
    if callable(fn):
        return float(fn(action)), "env.control_cost(action)"
    raise EvaluationError(
        f"{env.spec.id} reports no info['reward_ctrl'] and has no control_cost(); "
        "refusing to guess a control-cost formula")


def model_policy(model) -> PolicyFn:
    """Deterministic policy function from an SB3 model."""
    def act(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act


def run_episode(env, policy: PolicyFn, seed: int, max_steps: int,
                frame_sink: Callable[[np.ndarray], None] | None = None) -> dict[str, Any]:
    """Roll out one deterministic episode and return its metrics."""
    obs, reset_info = env.reset(seed=int(seed))
    start_x, x_source = _x_position(env, reset_info)
    end_x = start_x
    total_reward = 0.0
    ctrl_total = 0.0
    ctrl_source = None
    steps = 0
    done = False
    if frame_sink is not None:
        frame_sink(env.render())
    while not done and steps < max_steps:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        cost, ctrl_source = _control_cost(env, info, np.asarray(action))
        ctrl_total += cost
        end_x, _ = _x_position(env, info)
        steps += 1
        if frame_sink is not None:
            frame_sink(env.render())
    episode = {
        "seed": int(seed),
        "reward": total_reward,
        "distance": float(end_x - start_x),
        "control_cost": ctrl_total,
        "control_cost_per_step": ctrl_total / max(steps, 1),
        "length": steps,
        "initial_x_source": x_source,
        "control_cost_source": ctrl_source,
    }
    bad = [k for k in ("reward", "distance", "control_cost") if not math.isfinite(episode[k])]
    if bad:
        raise EvaluationError(f"non-finite episode metrics {bad} (seed {seed}): {episode}")
    return episode


def evaluate_policy(policy: PolicyFn, env_id: str = DEFAULT_ENV_ID, num_episodes: int = 10,
                    max_steps: int = 1000, base_seed: int = 0) -> dict[str, Any]:
    """Standardised evaluation. Episode ``i`` uses seed ``base_seed + i``."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    env = make_env(env_id)
    try:
        episodes = [run_episode(env, policy, base_seed + i, max_steps) for i in range(num_episodes)]
    finally:
        env.close()
    rewards = [e["reward"] for e in episodes]
    distances = [e["distance"] for e in episodes]
    costs = [e["control_cost"] for e in episodes]
    costs_per_step = [e["control_cost_per_step"] for e in episodes]
    return {
        "env_id": env_id,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "rewards": rewards,
        "episode_lengths": [e["length"] for e in episodes],
        "eval_seeds": [e["seed"] for e in episodes],
        "behavior": {
            "mean_distance": float(np.mean(distances)),
            "mean_control_cost": float(np.mean(costs)),
            "mean_control_cost_per_step": float(np.mean(costs_per_step)),
            "distances": distances,
            "control_costs": costs,
            "control_costs_per_step": costs_per_step,
            "initial_x_source": episodes[0]["initial_x_source"],
            "control_cost_source": episodes[0]["control_cost_source"],
        },
    }


def record_video(policy: PolicyFn, out_path: str | os.PathLike, env_id: str = DEFAULT_ENV_ID,
                 seed: int = 0, max_steps: int = 1000, fps: int | None = None) -> dict[str, Any]:
    """Render one deterministic episode to an MP4 (headless-safe).

    Rendering uses ``render_mode="rgb_array"``; on a headless machine set
    ``MUJOCO_GL=egl`` (GPU) or ``MUJOCO_GL=osmesa`` (CPU). Defaults to egl.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    env = make_env(env_id, render_mode="rgb_array")
    fps = fps or int(env.metadata.get("render_fps", 20))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7)
    try:
        episode = run_episode(env, policy, seed, max_steps, frame_sink=writer.append_data)
    finally:
        writer.close()
        env.close()
    episode["video_path"] = str(out_path)
    return episode


def _load_model(path: str, device: str):
    from stable_baselines3 import PPO
    if not os.path.exists(path):
        raise FileNotFoundError(f"trained model not found: {path}")
    return PPO.load(path, device=device)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deterministic MESB evaluation of a trained PPO model")
    p.add_argument("--model", required=True, help="Path to a saved SB3 PPO .zip")
    p.add_argument("--env", default=DEFAULT_ENV_ID)
    p.add_argument("--seed", type=int, default=1000, help="Base eval seed; episode i uses seed+i")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--video", default=None, help="Optional MP4 path (renders episode 0's seed)")
    p.add_argument("--json-out", default=None, help="Optional path for the metrics JSON")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    model = _load_model(args.model, args.device)
    policy = model_policy(model)
    result = evaluate_policy(policy, args.env, args.episodes, args.max_steps, args.seed)
    b = result["behavior"]
    print(f"Env: {args.env}  episodes: {args.episodes}  seeds: {args.seed}..{args.seed + args.episodes - 1}")
    for i, (r, d, c) in enumerate(zip(result["rewards"], b["distances"], b["control_costs"])):
        print(f"  episode {i:2d}: reward {r:10.2f}  distance {d:9.2f}  control_cost {c:9.3f}")
    print(f"mean_reward {result['mean_reward']:.2f} (std {result['std_reward']:.2f})  "
          f"mean_distance {b['mean_distance']:.2f}  mean_control_cost {b['mean_control_cost']:.3f}  "
          f"per_step {b['mean_control_cost_per_step']:.5f}")
    if args.video:
        ep = record_video(policy, args.video, args.env, seed=args.seed, max_steps=args.max_steps)
        result["video"] = ep
        print(f"Video written to {ep['video_path']} (reward {ep['reward']:.2f})")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
