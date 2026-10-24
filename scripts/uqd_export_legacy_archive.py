"""Export a legacy_baseline checkpoint into the uncertain-QD snapshot format.

The legacy pipeline keeps one aggregate (the 10-episode mean) per gene, so the
exported search estimates have n = 10 but no per-episode samples; they are
flagged ``aggregate_only``. Their SD is the legacy ``std_reward`` (population
SD, ddof = 0), not comparable to the engine's ddof = 1 SD, and is labelled so.
The held-out final evaluation then scores the archive exactly like an engine
run (scripts/uqd_final_eval.py --run-dir <out-dir>).

    uv run --isolated --project sota/MujocoRL/eval_env --with deap python \\
        scripts/uqd_export_legacy_archive.py --checkpoint <ckpt_dir>/checkpoint_gen_N.pkl \\
        --global-data <global_dir>/global_gen_N.pkl --out-dir mujoco_rl_output/uqd/legacy_seed0
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

from deap import base, creator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.cfg.uqd_presets import resolve_uqd_config  # noqa: E402
from src.uqd.archive import Grid  # noqa: E402
from src.uqd.budget import CostLedger  # noqa: E402
from src.uqd.engine import git_revision  # noqa: E402
from src.uqd.mujoco import BINS, DESCRIPTORS, evaluation_protocol  # noqa: E402

# Same definitions as run_improved.py, so its pickles load.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--global-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=int(os.getenv("MUJOCO_EVAL_SEED", "0")),
                        help="MUJOCO_EVAL_SEED the legacy run used")
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "sota/MujocoRL/trained_models"))
    parser.add_argument("--stats-dir", default=os.path.join(REPO_ROOT, "sota/MujocoRL/stats"))
    parser.add_argument("--env", default="Walker2d-v5")
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    with open(args.global_data, "rb") as f:
        gdata = pickle.load(f)
    metrics_store = {}
    for store in ("GLOBAL_DATA_HIST", "GLOBAL_DATA"):
        for gene, entry in gdata.get(store, {}).items():
            if entry.get("metrics"):
                metrics_store[gene] = entry["metrics"]

    grid = Grid(DESCRIPTORS, BINS)
    archive, elites, missing = [], {}, []
    for cell, ind in sorted(ckpt.get("MAP_ELITES_ARCHIVE", {}).items()):
        gene = ind[0]
        m = metrics_store.get(gene)
        if m is None:
            raise SystemExit(f"no metrics recorded for archived gene {gene}")
        model_path = os.path.join(args.models_dir, f"{gene}_seed{args.seed}.zip")
        sha = sha256(model_path) if os.path.exists(model_path) else None
        if sha is None:
            missing.append(model_path)
        sid = f"{gene}@t{args.seed}"
        desc = [m[name] for name, _, _ in DESCRIPTORS]
        assert tuple(cell) == grid.cell_of(desc), f"{gene}: archive cell disagrees with its descriptors"
        archive.append({
            "cell": list(cell), "rank": 0, "solution": sid, "gene": gene, "kind": "policy",
            "status": "eligible", "model_path": model_path, "checkpoint_sha256": sha,
            "param_count": m.get("param_count"),
            "estimate": {"n": args.eval_episodes, "fitness_mean": m["mean_reward"],
                         "fitness_sd": None, "fitness_sem": None, "fitness_median": None,
                         "descriptor_mean": desc, "descriptor_var": None, "descriptor_median": None,
                         "aggregate_only": True, "legacy_std_reward_ddof0": m.get("std_reward")},
        })
        elites[",".join(map(str, cell))] = sid

    # Search cost from the per-gene stats files that exist; everything else is unknown.
    ledger = CostLedger()
    for gene in sorted(set(gdata.get("GLOBAL_DATA_HIST", {})) | set(gdata.get("GLOBAL_DATA", {}))):
        path = os.path.join(args.stats_dir, f"{gene}_seed{args.seed}_stats.json")
        if not os.path.exists(path):
            ledger.record("initial_train", gene, 0, status="unknown", note="no stats file")
            continue
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        ok = st.get("status", "ok") == "ok" and "rewards" in st
        ledger.record("initial_train", gene, 0, env_steps=st.get("timesteps") if ok else None,
                      wall_seconds=st.get("train_wall_sec", st.get("train_time_sec")),
                      status="ok" if ok else "failed")
        if ok:
            ledger.record("initial_eval", gene, 0, env_steps=st.get("eval_env_steps"),
                          episodes=len(st["rewards"]), wall_seconds=st.get("eval_wall_sec"))

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = resolve_uqd_config("legacy_baseline")
    search_seeds = [args.seed] + [args.seed + k for k in range(args.eval_episodes)]
    snapshot = {"preset": "legacy_baseline", "strategy": "legacy_map_elites", "generation": None,
                "grid": {"descriptors": grid.descriptors, "bins": grid.bins}, "archive": archive,
                "elites": elites, "search_seeds_used": search_seeds,
                "illusory_metrics": {"filled_cells": len(archive), "coverage": len(archive) / grid.n_cells},
                "missing_checkpoints": missing}
    meta = {"preset": "legacy_baseline", "category": "legacy", "resolved_config": cfg,
            "seeds": {"run_seed": None, "legacy_eval_seed": args.seed, "final_master_seed":
                      cfg["final_eval"]["master_seed"]},
            "evaluation_protocol": evaluation_protocol(args.env, args.eval_max_steps),
            "exported_from": {"checkpoint": os.path.abspath(args.checkpoint),
                              "global_data": os.path.abspath(args.global_data)},
            "code_revision": git_revision(REPO_ROOT), "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "legacy estimates are 10-episode aggregates with shared reset seeds seed+0..9"}
    for name, obj in (("archive_latest.json", snapshot), ("run_metadata.json", meta)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)
    with open(os.path.join(args.out_dir, "cost_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger.to_list(), f)
    print(f"exported {len(archive)} elites ({len(missing)} missing checkpoints) to {args.out_dir}")


if __name__ == "__main__":
    main()
