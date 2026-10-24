"""Held-out final evaluation of a finished uncertain-QD (or exported legacy) run.

    uv run --isolated --project sota/MujocoRL/eval_env python scripts/uqd_final_eval.py \\
        --run-dir mujoco_rl_output/uqd/<preset>_seed<R> [--episodes 64] [--slurm]

Writes <run-dir>/final_eval/final_eval.json. See src/uqd/final_eval.py for
the metrics and src/uqd/final_runner.py for the inputs it expects.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.uqd.executors import LocalExecutor, SlurmExecutor  # noqa: E402
from src.uqd.final_runner import run_final_evaluation  # noqa: E402

CPU_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=uqdFinalEval
#SBATCH -t 4:00:00
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -N 1
#SBATCH --output={output}
hostname
module load uv
unset VIRTUAL_ENV
cd "$SLURM_SUBMIT_DIR"
{command}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=None, help="held-out episodes per checkpoint")
    parser.add_argument("--snapshot", default="archive_latest.json")
    parser.add_argument("--slurm", action="store_true", help="submit rollout chunks with sbatch")
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    work = os.path.join(os.path.abspath(args.run_dir), "final_eval", "jobs")
    if args.slurm:
        executor = SlurmExecutor(REPO_ROOT, work, CPU_TEMPLATE, chunk_size=args.chunk_size)
    else:
        executor = LocalExecutor(REPO_ROOT, work)
    out = run_final_evaluation(args.run_dir, executor, episodes=args.episodes, snapshot_name=args.snapshot)
    for name, rep in out["metrics"]["reports"].items():
        print(name, json.dumps({k: v for k, v in rep.items() if k != "corrected_archive"}, default=str))
    print("Job Done")


if __name__ == "__main__":
    main()
