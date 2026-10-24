"""Post-hoc ARIA refinement of a finished run (results go to <run-dir>/aria/).

    uv run --isolated --project sota/MujocoRL/eval_env python scripts/uqd_aria.py \\
        --run-dir mujoco_rl_output/uqd/<preset>_seed<R> [--final-episodes 64] [--slurm]

Uses the run's resolved "aria" settings unless --aria-json overrides them
(e.g. a smaller smoke-test budget). Every budget below the paper's is
labelled aria_reduced_budget in the outputs.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.cfg.uqd_presets import DEFAULTS, validate_uqd_config  # noqa: E402
from src.uqd.aria_runner import run_aria_stage  # noqa: E402
from src.uqd.executors import LocalExecutor, SlurmExecutor  # noqa: E402

ARIA_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=uqdAria
#SBATCH -t 8:00:00
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -N 1
#SBATCH --output={output}
hostname
module load uv
unset VIRTUAL_ENV
{command}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--final-episodes", type=int, default=None)
    parser.add_argument("--post-eval-episodes", type=int, default=8)
    parser.add_argument("--aria-json", default="", help="JSON merged over the run's ARIA settings")
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "run_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = dict(meta["resolved_config"]["aria"])
    if args.aria_json:
        cfg.update(json.loads(args.aria_json))
        check = json.loads(json.dumps(meta["resolved_config"]))
        check["aria"] = cfg
        unknown = set(cfg) - set(DEFAULTS["aria"])
        if unknown:
            raise SystemExit(f"unknown ARIA settings: {sorted(unknown)}")
        validate_uqd_config(check)
    work = os.path.join(os.path.abspath(args.run_dir), "aria", "jobs")
    executor = (SlurmExecutor(REPO_ROOT, work, ARIA_TEMPLATE, timeout=48 * 3600) if args.slurm
                else LocalExecutor(REPO_ROOT, work))
    out = run_aria_stage(args.run_dir, final_episodes=args.final_episodes, executor=executor,
                         post_eval_episodes=args.post_eval_episodes, aria_cfg=cfg)
    print(json.dumps(out["summary"], indent=1, default=str))
    print("Job Done")


if __name__ == "__main__":
    main()
