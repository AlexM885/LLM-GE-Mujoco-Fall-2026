"""
Learning curve of one seed config (PPO or TD3) on HalfCheetah-v4, for comparing
the three LLM-GE trainer configs against XQC under the same protocol.

Trains the gene for -timesteps env steps and every -eval_every steps runs
-eval_episodes deterministic episodes on a separately seeded env. One row per
eval is appended to <out_dir>/eval.csv (same layout as sota/XQC eval.csv plus the
MAP-Elites descriptors), so a job killed at the wall limit still leaves a curve.

    python baseline_curve.py -algo ppo -gene network.py -seed 0 -out_dir runs/ppo_s0
    python baseline_curve.py -algo td3 -gene seeds/td3_network.py -seed 0 -out_dir runs/td3_s0
"""

import os
import csv
import json
import time
import argparse
import importlib.util

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from eval import evaluate_model

EVAL_FIELDS = ["algo", "env", "seed", "env_step", "return", "return_std",
               "mean_distance", "mean_control_cost", "wall_time_s", "slurm_job_id"]


def load_gene(path):
    spec = importlib.util.spec_from_file_location("gene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(algo, gene, env, seed, timesteps):
    policy_kwargs = gene.get_policy_kwargs()
    policy_class = policy_kwargs.pop("policy_class", "MlpPolicy")
    if algo == "ppo":
        kwargs = gene.get_ppo_kwargs()
        # SB3 recommends CPU for PPO with MLP policies.
        return PPO(policy_class, env, policy_kwargs=policy_kwargs, seed=seed, device="cpu", **kwargs)

    kwargs = gene.get_td3_kwargs()
    # Gaussian exploration noise, as in Alex's train_rl.py (8d4c0b2).
    exploration_noise = float(kwargs.pop("exploration_noise", 0.1))
    if exploration_noise > 0 and "action_noise" not in kwargs:
        n_actions = env.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=exploration_noise * np.ones(n_actions))
    # Smoke runs: make sure some gradient updates happen.
    if kwargs.get("learning_starts", 100) >= timesteps:
        kwargs["learning_starts"] = max(timesteps // 2, 1)
    return TD3(policy_class, env, policy_kwargs=policy_kwargs, seed=seed, **kwargs)


class EvalLogger(BaseCallback):
    def __init__(self, log_eval, eval_every):
        super().__init__()
        self.log_eval = log_eval
        self.eval_every = eval_every
        self.next_eval = eval_every

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.log_eval(self.model, self.num_timesteps)
            self.next_eval += self.eval_every
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-algo", choices=["ppo", "td3"], required=True)
    parser.add_argument("-gene", required=True, help="Path to the gene .py file")
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-env", default="HalfCheetah-v4")
    parser.add_argument("-timesteps", type=int, default=1_000_000)
    parser.add_argument("-eval_every", type=int, default=25_000)
    parser.add_argument("-eval_episodes", type=int, default=10)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, "eval.csv")
    if os.path.exists(os.path.join(args.out_dir, "DONE")):
        print(f"[done] {args.out_dir}/DONE exists, nothing to do", flush=True)
        return
    # No resume: a rerun starts the curve over.
    with open(eval_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=EVAL_FIELDS).writeheader()

    gene = load_gene(args.gene)
    gene_algo = getattr(gene, "ALGO", args.algo)
    if gene_algo != args.algo:
        raise SystemExit(f"gene {args.gene} declares ALGO={gene_algo!r}, not {args.algo!r}")

    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    eval_env = gym.make(args.env)
    eval_env.reset(seed=args.seed + 42)

    model = build_model(args.algo, gene, env, args.seed, args.timesteps)
    if args.algo == "td3":
        actor_params = sum(p.numel() for p in model.actor.parameters())
    else:
        p = model.policy
        actor_params = (sum(t.numel() for t in p.mlp_extractor.policy_net.parameters())
                        + sum(t.numel() for t in p.action_net.parameters()) + p.log_std.numel())
    start = time.time()
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    def log_eval(model, env_step):
        mean_r, std_r, _, metrics = evaluate_model(
            model, eval_env, num_episodes=args.eval_episodes, max_steps=1000)
        row = {
            "algo": args.algo, "env": args.env, "seed": args.seed, "env_step": env_step,
            "return": round(mean_r, 2), "return_std": round(std_r, 2),
            "mean_distance": round(metrics["mean_distance"], 3),
            "mean_control_cost": round(metrics["mean_control_cost"], 5),
            "wall_time_s": round(time.time() - start, 1), "slurm_job_id": job_id,
        }
        with open(eval_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=EVAL_FIELDS).writerow(row)
        print(f"[eval] env_step={env_step} return={mean_r:.1f} +- {std_r:.1f} "
              f"wall={row['wall_time_s']}s", flush=True)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "actor_params": actor_params,
                   "total_params": sum(p.numel() for p in model.policy.parameters())}, f, indent=2)

    model.learn(total_timesteps=args.timesteps, callback=EvalLogger(log_eval, args.eval_every))
    if model.num_timesteps % args.eval_every != 0:
        log_eval(model, model.num_timesteps)
    open(os.path.join(args.out_dir, "DONE"), "w").close()
    print(f"[done] {args.timesteps} steps in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
