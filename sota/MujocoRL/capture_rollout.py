"""Capture ordered frames + telemetry from a trained PPO checkpoint.

This is the rollout half of the sparse-visual-feedback method. It runs inside
the dedicated MuJoCo evaluation environment (``sota/MujocoRL/eval_env``), which
already provides gymnasium/mujoco/stable-baselines3/imageio but no pyyaml or
transformers. It deliberately does **not** import the repository constants; the
observer job (root environment) is responsible for turning a capture into a
cached observation and for computing the behaviour cache key.

Standalone usage::

    uv run --isolated --project sota/MujocoRL/eval_env python \
        sota/MujocoRL/capture_rollout.py \
        --model sota/MujocoRL/trained_models/network_<gene>.zip \
        --gene-id <gene> \
        --out-dir sota/MujocoRL/behavior_captures/<gene>

Frame selection: for each rollout we keep a short contiguous window from the
start and a short contiguous window ending at termination, so temporal order is
meaningful and a failing episode still contributes a pre-termination window.
The number of rendered frames is bounded by ``render_stride`` regardless of the
episode length.
"""

import argparse
import collections
import datetime
import json
import math
import os
import time

import numpy as np


def _downscale(frame, max_side):
    """Block-average an HxWx3 uint8 frame so its longest side is <= max_side."""
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or max_side <= 0:
        return frame.astype(np.uint8)
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame.astype(np.uint8)
    step = int(math.ceil(longest / float(max_side)))
    if step <= 1:
        return frame.astype(np.uint8)
    hh = (height // step) * step
    ww = (width // step) * step
    if hh == 0 or ww == 0:
        return frame.astype(np.uint8)
    block = frame[:hh, :ww].astype(np.float32)
    block = block.reshape(hh // step, step, ww // step, step, 3).mean(axis=(1, 3))
    return np.clip(block, 0, 255).astype(np.uint8)


def _render(env, camera=None):
    """Render one RGB frame, selecting a named camera on the live renderer.

    Gymnasium creates ``mujoco_renderer`` lazily on the first ``render()`` call
    and only accepts ``camera_name`` at construction, so the camera is applied to
    the already-created renderer here (see ``main`` for the initialising render).
    """
    if camera:
        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        if renderer is not None:
            try:
                renderer.camera_name = camera
            except Exception:
                pass
    try:
        return env.render()
    except Exception:
        return None


def _torso_state(env):
    """(z, pitch) for Walker2d's qpos layout, or (None, None) on other robots."""
    try:
        qpos = np.asarray(env.unwrapped.data.qpos)
        return float(qpos[1]), float(qpos[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None, None


def _episode_summary(ep_index, seed, total_reward, steps, terminated, start_x,
                     end_x, z_values, pitch_values, contact_steps,
                     ctrl_cost, foot_names, terms):
    # ep_index is the 0-based episode order; seed is the reset seed.
    def mean(values):
        return float(np.mean(values)) if values else 0.0

    summary = {
        "index": ep_index,
        "seed": seed,
        "return": float(total_reward),
        "steps": int(steps),
        "terminated": bool(terminated),
        "termination_step": int(steps) if terminated else None,
        "distance": float(end_x - start_x) if (start_x is not None and end_x is not None) else 0.0,
        "mean_torso_z": mean(z_values),
        "final_torso_z": float(z_values[-1]) if z_values else 0.0,
        "final_pitch": float(pitch_values[-1]) if pitch_values else 0.0,
        "mean_control_cost": float(ctrl_cost) / max(steps, 1),
    }
    for i, grounding in enumerate(contact_steps):
        summary[f"foot_contact_{i}"] = float(grounding) / max(steps, 1)
    summary["foot_geom_names"] = foot_names
    if terms is not None:
        summary["return_forward"] = float(terms["forward"])
        summary["return_healthy"] = float(terms["healthy"])
        summary["return_ctrl"] = float(terms["ctrl"])
    return summary


def capture_episode(model, env, seed, ep_index=0, max_steps=1000, camera=None,
                    start_n=4, end_n=4, render_stride=5, max_side=336,
                    foot_geom_ids=None, feet_in_contact=None, get_x=None,
                    torso_state=None):
    """Run one deterministic rollout and return (frames, summary, series).

    frames: ordered list of (step, downscaled ndarray).
    series: downsampled per-step telemetry for the observer's text context.
    """
    if get_x is None:
        get_x = lambda env, info: None
    if torso_state is None:
        torso_state = _torso_state

    obs, reset_info = env.reset(seed=seed)
    start_x = get_x(env, reset_info)
    end_x = start_x
    start_frames = []
    end_ring = collections.deque(maxlen=max(0, end_n))
    series = []
    z_values, pitch_values = [], []
    ep_terms = {"forward": 0.0, "healthy": 0.0, "ctrl": 0.0}
    terms_valid = True
    contact_steps = np.zeros(len(foot_geom_ids) if foot_geom_ids else 0)
    total_reward = 0.0
    ctrl_cost = 0.0
    steps = 0
    terminated = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        terminated = bool(term)
        total_reward += float(reward)
        steps = step + 1
        ctrl_cost += float(np.sum(np.square(np.asarray(action, dtype=float))))

        current_x = get_x(env, info)
        if current_x is not None:
            end_x = current_x
        z, pitch = torso_state(env)
        if z is not None:
            z_values.append(z)
            pitch_values.append(pitch)

        r_fwd = info.get("reward_forward")
        r_alive = info.get("reward_survive")
        r_ctrl = info.get("reward_ctrl")
        if None not in (r_fwd, r_alive, r_ctrl):
            ep_terms["forward"] += float(r_fwd)
            ep_terms["healthy"] += float(r_alive)
            ep_terms["ctrl"] += float(r_ctrl)
        else:
            terms_valid = False

        if foot_geom_ids and feet_in_contact is not None:
            contact_steps += np.array(feet_in_contact(env, foot_geom_ids), dtype=float)

        if step % max(1, render_stride) == 0:
            frame = None
            try:
                frame = _downscale(_render(env, camera), max_side)
            except Exception:
                frame = None
            if frame is not None:
                if len(start_frames) < max(0, start_n):
                    start_frames.append((step, frame))
                elif end_ring.maxlen:
                    end_ring.append((step, frame))
            series.append({
                "step": step,
                "x": current_x,
                "torso_z": z,
                "pitch": pitch,
                "action_norm": float(np.linalg.norm(np.asarray(action, dtype=float))),
                "reward": float(reward),
            })

        if done:
            break

    frames = start_frames + list(end_ring)
    frames.sort(key=lambda item: item[0])
    foot_names = []
    summary = _episode_summary(
        ep_index=ep_index, seed=seed, total_reward=total_reward, steps=steps,
        terminated=terminated, start_x=start_x, end_x=end_x, z_values=z_values,
        pitch_values=pitch_values, contact_steps=contact_steps, ctrl_cost=ctrl_cost,
        foot_names=foot_names, terms=ep_terms if terms_valid else None)
    return frames, summary, series


def save_capture(out_dir, all_frames, telemetry, manifest):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    import imageio.v2 as imageio

    frame_index = []
    for order, (episode, step, frame) in enumerate(all_frames):
        name = f"frame_{order:04d}.png"
        imageio.imwrite(os.path.join(frames_dir, name), frame)
        frame_index.append({"index": order, "episode": episode, "step": step,
                            "file": f"frames/{name}"})

    telemetry["frames"] = frame_index
    with open(os.path.join(out_dir, "telemetry.json"), "w", encoding="utf-8") as handle:
        json.dump(telemetry, handle, indent=2, sort_keys=True)

    manifest["num_frames"] = len(frame_index)
    manifest["frame_index"] = frame_index
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return frames_dir


def _versions():
    versions = {}
    for name in ("gymnasium", "mujoco", "stable_baselines3", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def main():
    parser = argparse.ArgumentParser(
        description="Capture ordered frames and telemetry from a trained checkpoint")
    parser.add_argument("--model", required=True, help="Path to trained .zip checkpoint")
    parser.add_argument("--gene-id", required=True, help="Parent gene id")
    parser.add_argument("--out-dir", required=True, help="Output capture directory")
    parser.add_argument("--genome", default=None, help="Optional parent .py path to record")
    parser.add_argument("--env", default=os.getenv("MUJOCO_ENV_ID", "Walker2d-v5"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera", default="track")
    parser.add_argument("--image-max-side", type=int, default=336)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--render-stride", type=int, default=5)
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing the review mp4")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval import (make_env, validate_reward_spec, get_foot_geom_ids,
                      _feet_in_contact, _get_x_position)
    from stable_baselines3 import PPO

    if not os.path.isfile(args.model):
        raise SystemExit(f"checkpoint not found: {args.model}")

    start_time = time.time()
    env = make_env(args.env, render_mode="rgb_array")
    for problem in validate_reward_spec(env):
        print(f"  WARNING: fitness spec mismatch -> {problem}")
    model = PPO.load(args.model, env=env)
    foot_ids, foot_names = get_foot_geom_ids(env)
    # Initialise the MuJoCo renderer once so the named camera can be applied,
    # and fail loudly here rather than silently capturing no frames.
    try:
        first = env.render()
        if first is None:
            print("  WARNING: env.render() returned None; capture will be empty")
    except Exception as exc:
        print(f"  WARNING: initial render failed ({exc})")

    start_n = max(0, args.frames // (2 * max(1, args.episodes)))
    end_n = max(0, args.frames // max(1, args.episodes) - start_n)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    print(f"Capturing {args.episodes} episodes, seeds={seeds}, "
          f"start_window={start_n}, end_window={end_n}")

    all_frames = []
    episodes = []
    for index, seed in enumerate(seeds):
        frames, summary, series = capture_episode(
            model, env, seed=seed, ep_index=index, max_steps=args.max_steps,
            camera=args.camera,
            start_n=start_n, end_n=end_n, render_stride=args.render_stride,
            max_side=args.image_max_side, foot_geom_ids=foot_ids,
            feet_in_contact=_feet_in_contact, get_x=_get_x_position)
        summary["foot_geom_names"] = foot_names
        summary["series"] = series
        episodes.append(summary)
        for step, frame in frames:
            all_frames.append((index, step, frame))
        print(f"  episode {index}: return={summary['return']:.2f} "
              f"steps={summary['steps']} frames={len(frames)}")
    env.close()

    if not all_frames:
        raise SystemExit("no frames captured; check render_mode/camera availability")

    telemetry = {"episodes": episodes, "camera": args.camera,
                 "image_max_side": args.image_max_side}
    manifest = {
        "parent_gene": args.gene_id,
        "model": os.path.abspath(args.model),
        "genome": os.path.abspath(args.genome) if args.genome else None,
        "env_id": args.env,
        "episodes": args.episodes,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "frames_target": args.frames,
        "camera": args.camera,
        "image_max_side": args.image_max_side,
        "render_stride": args.render_stride,
        "start_window": start_n,
        "end_window": end_n,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "versions": _versions(),
        "capture_seconds": round(time.time() - start_time, 3),
    }
    frames_dir = save_capture(args.out_dir, all_frames, telemetry, manifest)

    if not args.no_video:
        try:
            import imageio.v2 as imageio
            video_path = os.path.join(args.out_dir, "rollout.mp4")
            imageio.mimsave(video_path, [f for _, _, f in all_frames], fps=10)
            print(f"Saved review video: {video_path}")
        except Exception as exc:
            print(f"  WARNING: could not write review video ({exc})")

    print(f"Saved {len(all_frames)} frames to {frames_dir}")
    print("Job Done")


if __name__ == "__main__":
    main()
