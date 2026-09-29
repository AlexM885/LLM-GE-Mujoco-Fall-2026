"""Train one LLM-generated PPO gene on HalfCheetah and evaluate it for MESB.

This is the MESB counterpart of ``train_rl.py`` on the reference branches.
It keeps the same seed-network contract (``get_policy_kwargs()`` returning
``{"policy_class": GenePolicy}`` and ``get_ppo_kwargs()``), but:

* the gene is loaded from an explicit file path (no shared ``models/`` dir),
* training and evaluation are seeded separately,
* evaluation is the standardised deterministic protocol in ``behavior_eval``,
* the canonical output is a named JSON file; failures always produce
  ``"status": "failed"`` with the stage and error.

Example (smoke test, ~1 minute on a laptop CPU)::

    python sota/MujocoRL/train_gene.py --gene-file sota/MujocoRL/network.py \
        --gene-id seed_smoke --out-json /tmp/seed_smoke.json \
        --model-dir /tmp/models --timesteps 2048 --eval-episodes 2 --eval-max-steps 200
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from behavior_eval import DEFAULT_ENV_ID, evaluate_policy, make_env, model_policy  # noqa: E402
from sota.MujocoRL.results_io import (SCHEMA_VERSION, atomic_write_json,  # noqa: E402
                                      failure_payload, validate_payload)


def load_gene_module(gene_file: str | os.PathLike, gene_id: str):
    """Import an LLM-generated network file from an explicit path."""
    gene_file = Path(gene_file)
    if not gene_file.exists():
        raise FileNotFoundError(f"gene file not found: {gene_file}")
    spec = importlib.util.spec_from_file_location(f"mesb_gene_{gene_id}", gene_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn in ("get_policy_kwargs", "get_ppo_kwargs"):
        if not callable(getattr(module, fn, None)):
            raise AttributeError(f"gene {gene_id} does not define {fn}()")
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def clamp_for_small_runs(ppo_kwargs: dict, timesteps: int) -> dict:
    """Same small-run clamp as the reference train_rl.py (smoke tests only)."""
    if timesteps <= 2048:
        ppo_kwargs["n_steps"] = min(ppo_kwargs.get("n_steps", 2048), max(timesteps, 64))
        ppo_kwargs["batch_size"] = min(ppo_kwargs.get("batch_size", 64), ppo_kwargs["n_steps"])
    return ppo_kwargs


def run(args: argparse.Namespace) -> dict:
    start = time.time()
    base = {
        "schema_version": SCHEMA_VERSION,
        "gene_id": args.gene_id,
        "generation": args.generation,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "env_id": args.env,
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "num_eval_episodes": args.eval_episodes,
        "max_eval_steps": args.eval_max_steps,
        "gene_file": str(args.gene_file),
    }
    stage = "load_gene"
    try:
        module = load_gene_module(args.gene_file, args.gene_id)
        policy_kwargs = dict(module.get_policy_kwargs())
        ppo_kwargs = dict(module.get_ppo_kwargs())
        policy_class = policy_kwargs.pop("policy_class", "MlpPolicy")

        stage = "build_model"
        from stable_baselines3 import PPO
        seed_everything(args.seed)
        env = make_env(args.env)
        ppo_kwargs = clamp_for_small_runs(ppo_kwargs, args.timesteps)
        ppo_kwargs.pop("seed", None)
        ppo_kwargs.pop("device", None)
        model = PPO(policy=policy_class, env=env, policy_kwargs=policy_kwargs,
                    seed=args.seed, device=args.device, **ppo_kwargs)
        # Same convention as the reference train_rl.py (PPO): actor + critic.
        param_count = int(sum(p.numel() for p in model.policy.parameters()))

        stage = "train"
        model.learn(total_timesteps=args.timesteps)
        train_time = time.time() - start
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        model_path = str(Path(args.model_dir) / f"{args.gene_id}.zip")
        model.save(model_path)
        env.close()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"trained model was not written: {model_path}")

        stage = "evaluate"
        result = evaluate_policy(model_policy(model), args.env, args.eval_episodes,
                                 args.eval_max_steps, args.eval_seed)
        payload = dict(base)
        payload.update({
            "status": "success",
            "train_time_sec": train_time,
            "total_time_sec": time.time() - start,
            "mean_reward": result["mean_reward"],
            "std_reward": result["std_reward"],
            "param_count": param_count,
            "model_path": model_path,
            "rewards": result["rewards"],
            "episode_lengths": result["episode_lengths"],
            "eval_seeds": result["eval_seeds"],
            "behavior": result["behavior"],
        })
        ok, reason = validate_payload(payload, args.gene_id)
        if not ok:  # e.g. NaN reward from a diverged policy
            stage = "validate"
            raise ValueError(f"invalid evaluation: {reason}")
        return payload
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded
        context = {k: v for k, v in base.items() if k != "gene_id"}
        return failure_payload(
            args.gene_id, f"{type(exc).__name__}: {exc}", **context, stage=stage,
            traceback=traceback.format_exc(), total_time_sec=time.time() - start)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train + evaluate one PPO gene for MESB")
    p.add_argument("--gene-file", required=True)
    p.add_argument("--gene-id", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--env", default=DEFAULT_ENV_ID)
    p.add_argument("--seed", type=int, default=0, help="PPO training seed")
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="Base evaluation seed (episode i uses eval_seed + i)")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-max-steps", type=int, default=1000)
    p.add_argument("--device", default="cpu",
                   help="SB3 device. PPO+MLP is usually fastest (and more reproducible) on CPU")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    atomic_write_json(args.out_json, payload)
    if payload["status"] == "success":
        b = payload["behavior"]
        print(f"[{args.gene_id}] mean_reward {payload['mean_reward']:.2f} "
              f"std {payload['std_reward']:.2f} distance {b['mean_distance']:.2f} "
              f"control_cost {b['mean_control_cost']:.3f} params {payload['param_count']} "
              f"train {payload['train_time_sec']:.1f}s")
        print("Job Done")
        return 0
    print(f"[{args.gene_id}] FAILED at stage {payload.get('stage')}: {payload['error']}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
