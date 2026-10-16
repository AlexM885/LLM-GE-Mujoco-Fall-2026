"""Capture an XQC-trained actor's rollout for the VLM observer.

Rolls out a single seed of a saved XQC checkpoint deterministically on
HalfCheetah-v4 (or any Gymnasium Mujoco env XQC was trained on), and writes
the same ``frames/`` + ``telemetry.json`` layout that ``capture_rollout.py``
produces for SB3 policies. This is the bridge piece that unlocks multi-
generation evolution: trained XQC policies -> frames -> VLM observer ->
new aux_reward_cache.json -> next-gen training.

Usage:
    python capture_xqc_rollout.py \
        --run-dir /storage/.../xqc_vlm_fwdvel_lam10_a_runs/xqc_pilot_seed \
        --seed-index 7                       # which parallel seed to roll out
        --env HalfCheetah-v4 \
        --out-dir /storage/.../captures/hc_gen1_best \
        --episodes 3 \
        --max-steps 1000 \
        --target-frames 32

Requires: XQC venv (JAX + flax + the XQC package on PYTHONPATH).
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Make sure the XQC package resolves whether we run from /sota/XQC or elsewhere.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llmge_descriptors import (  # type: ignore
    make_env, get_foot_geom_ids, feet_in_contact, _x_position, _forward_reward,
)


# --------------------------------------------------------------------------- #
# Agent construction + checkpoint load
# --------------------------------------------------------------------------- #

def build_agent_from_run_dir(run_dir, env_name):
    """Rebuild the XQC learner matching the agent-config found in run_dir and
    load its latest checkpoint. Mirrors the agent construction in train_xqc.py.
    """
    from omegaconf import OmegaConf
    from xqc import checkpoint as xqc_ckpt
    from xqc.agents.xqc.xqc_learner import XQCLearner

    cfg_path = next((run_dir / "hydra").rglob(".hydra/config.yaml"), None)
    if cfg_path is None:
        # train_xqc also writes a flat config.json copy at the root.
        cfg_json = run_dir / "config.json"
        if not cfg_json.exists():
            raise FileNotFoundError(
                f"No hydra config.yaml or config.json under {run_dir}; cannot rebuild agent.")
        cfg = OmegaConf.create(json.load(open(cfg_json)))
    else:
        cfg = OmegaConf.load(cfg_path)

    env = make_env(env_name, seed=0)
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()

    num_seeds = int(cfg.get("num_seeds", 10))
    seed = int(cfg.get("seed", 0))

    # Instantiate the learner with the same shape as training; we won't call
    # update() so critic/temperature/buffer configs only need to type-check.
    agent = XQCLearner(
        seed, obs_space.sample()[None], act_space.sample()[None],
        num_seeds=num_seeds, **OmegaConf.to_container(cfg.agent, resolve=True))

    # Find latest checkpoint and load its agent-state leaves.
    ckpt_dir = xqc_ckpt.find_latest(run_dir)
    if ckpt_dir is None:
        raise FileNotFoundError(f"No complete checkpoint under {run_dir}/checkpoints")
    with open(Path(ckpt_dir) / "agent.pkl", "rb") as f:
        leaves = __import__("pickle").load(f)
    xqc_ckpt.load_agent_state(agent, leaves)
    print(f"[capture] loaded agent from {ckpt_dir} (num_seeds={num_seeds})", flush=True)
    return agent, num_seeds


# --------------------------------------------------------------------------- #
# Frame sampling (bookend: start_n at the beginning + end_n at the end)
# --------------------------------------------------------------------------- #

def _downscale(frame, max_side):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return frame
    scale = max_side / float(long_side)
    new_h, new_w = int(h * scale), int(w * scale)
    # cheap nearest-neighbour decimation; good enough for VLM input.
    y = np.linspace(0, h - 1, new_h).astype(np.int32)
    x = np.linspace(0, w - 1, new_w).astype(np.int32)
    return frame[y][:, x]


def _sample_bookend(start_frames, end_ring, target_total):
    if target_total <= 0:
        return []
    end_list = list(end_ring)
    picked = start_frames + end_list
    if len(picked) <= target_total:
        return picked
    half = target_total // 2
    return start_frames[:half] + end_list[-(target_total - half):]


# --------------------------------------------------------------------------- #
# Per-episode rollout
# --------------------------------------------------------------------------- #

def rollout_single_seed(agent, env_name, seed_index, episode_index,
                        reset_seed, max_steps, render_stride, max_side,
                        start_n, end_n):
    """Roll out one parallel-seed slice of the agent, recording frames +
    per-step telemetry for the VLM observer."""
    # XQC's agent evaluates all num_seeds in a batched vmap. We feed the same
    # observation into every slot but only record the one we care about.
    num_seeds = agent.actor.params["params"]["Dense_0"]["kernel"].shape[0] \
        if "params" in agent.actor.params else len(agent.seeds)

    env = make_env(env_name, seed=reset_seed)
    foot_ids, foot_names = get_foot_geom_ids(env)

    obs, reset_info = env.reset(seed=reset_seed)
    start_x = _x_position(env, reset_info)

    start_frames = []
    end_ring = collections.deque(maxlen=max(0, end_n))
    series = []
    contact_steps = np.zeros(len(foot_ids))
    ep_terms = {"forward": 0.0, "healthy": 0.0, "ctrl": 0.0}
    total_reward = 0.0
    steps = 0
    terminated = False
    end_x = start_x
    z_values, pitch_values = [], []
    try:
        qpos0 = np.asarray(env.unwrapped.data.qpos)
        z0, pitch0 = float(qpos0[0]), float(qpos0[1])
    except Exception:
        z0, pitch0 = None, None

    for step in range(max_steps):
        # Broadcast the single-seed observation across all parallel seeds.
        obs_broadcast = np.broadcast_to(obs.astype(np.float32),
                                        (num_seeds, obs.shape[0])).copy()
        actions_batch = np.asarray(agent.sample_actions(obs_broadcast, temperature=0.0))
        action = np.asarray(actions_batch[seed_index])

        new_obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        terminated = bool(term)
        total_reward += float(reward)
        steps = step + 1

        try:
            qpos = np.asarray(env.unwrapped.data.qpos)
            z_values.append(float(qpos[0]))
            pitch_values.append(float(qpos[1]))
        except Exception:
            pass

        current_x = _x_position(env, info)
        if current_x is not None:
            end_x = current_x
        for key, acc in (("reward_forward", "forward"),
                         ("reward_healthy", "healthy"),
                         ("reward_ctrl", "ctrl")):
            v = info.get(key)
            if v is not None:
                ep_terms[acc] += float(v)
        contact_steps += np.array(feet_in_contact(env, foot_ids), dtype=float)

        if step % max(1, render_stride) == 0:
            try:
                raw = env.render()
            except Exception:
                raw = None
            if raw is not None:
                frame = _downscale(raw, max_side)
                if frame is not None:
                    if len(start_frames) < start_n:
                        start_frames.append((step, frame))
                    end_ring.append((step, frame))
                    series.append({
                        "step": step,
                        "reward": float(reward),
                        "x_velocity": float(info.get("x_velocity", 0.0)),
                        "ctrl_cost": float(info.get("reward_ctrl", 0.0)) * -1.0,
                    })

        obs = new_obs
        if done:
            break

    env.close()

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    summary = {
        "index": int(episode_index),
        "seed": int(reset_seed),
        "return": float(total_reward),
        "steps": int(steps),
        "terminated": bool(terminated),
        "termination_step": int(steps) if terminated else None,
        "distance": float(end_x - start_x) if (start_x is not None and end_x is not None) else 0.0,
        "mean_torso_z": mean(z_values),
        "final_torso_z": float(z_values[-1]) if z_values else 0.0,
        "final_pitch": float(pitch_values[-1]) if pitch_values else 0.0,
        "mean_control_cost": float(-ep_terms["ctrl"] / max(steps, 1)),
        "return_forward": float(ep_terms["forward"]),
        "return_healthy": float(ep_terms["healthy"]),
        "return_ctrl": float(ep_terms["ctrl"]),
        "foot_geom_names": foot_names,
    }
    for i, g in enumerate(contact_steps):
        summary[f"foot_contact_{i}"] = float(g) / max(steps, 1)

    frames = _sample_bookend(start_frames, end_ring, start_n + end_n)
    return frames, summary, series


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Trained XQC run_dir with /checkpoints/latest")
    p.add_argument("--seed-index", type=int, required=True,
                   help="Which parallel seed (0..num_seeds-1) to roll out")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--render-stride", type=int, default=5)
    p.add_argument("--max-side", type=int, default=336)
    p.add_argument("--start-n", type=int, default=4)
    p.add_argument("--end-n", type=int, default=4)
    p.add_argument("--base-reset-seed", type=int, default=1000)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"; frames_dir.mkdir(exist_ok=True)

    # EGL headless for PACE compute nodes.
    os.environ.setdefault("MUJOCO_GL", "egl")

    agent, num_seeds = build_agent_from_run_dir(run_dir, args.env)
    if args.seed_index < 0 or args.seed_index >= num_seeds:
        raise ValueError(f"--seed-index {args.seed_index} out of range for num_seeds={num_seeds}")

    telemetry = {"episodes": [], "frame_manifest": [], "env": args.env,
                 "source": "capture_xqc_rollout.py", "seed_index": args.seed_index,
                 "run_dir": str(run_dir)}

    for ep in range(args.episodes):
        reset_seed = args.base_reset_seed + ep
        frames, summary, series = rollout_single_seed(
            agent, args.env, args.seed_index, ep, reset_seed,
            args.max_steps, args.render_stride, args.max_side,
            args.start_n, args.end_n)
        telemetry["episodes"].append(summary)
        telemetry.setdefault("series", {})[str(ep)] = series
        for step, frame in frames:
            name = f"ep{ep:02d}_step{step:05d}.png"
            imageio.imwrite(str(frames_dir / name), frame)
            telemetry["frame_manifest"].append({
                "episode": ep, "step": int(step),
                "file": f"frames/{name}",
            })
        print(f"[capture] ep={ep} seed_index={args.seed_index} "
              f"return={summary['return']:.1f} steps={summary['steps']} "
              f"saved {len(frames)} frames", flush=True)

    with open(out_dir / "telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[capture] wrote {len(telemetry['frame_manifest'])} frames + telemetry.json to {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
