"""Retrain one LLM-GE XQC gene and record gifs of the trained policy (for presentations).

To film a gene that evolution already trained, use render_actor.py (no retraining).
This script retrains the gene with the same settings as its evaluation (train_xqc_gene.py: same seeds, steps and validation)
and then saves one deterministic rollout gif per seed. It writes to its own run
dir and never touches llmge/results, so the evolution's scores stay as they were.

    python video_gene.py llmge/models/network_<gene_id>.py -timesteps 100000 -num_seeds 3
Output: $XQC_RUNS/llmge_video/<name>/{video/seed<s>.gif, video/returns.json, descriptors.json, train.log}
"""

import argparse
import os
import subprocess
import sys

from train_xqc_gene import HERE, gene_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gene", help="path to a gene file (llmge/models/network_<id>.py or llmge/network.py)")
    parser.add_argument("-name", help="run name (default: the gene id)")
    parser.add_argument("-timesteps", type=int, default=100_000)
    parser.add_argument("-num_seeds", type=int, default=3)
    parser.add_argument("-video_steps", type=int, default=300)
    parser.add_argument("-env", default="HalfCheetah-v4")
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    name = args.name or os.path.basename(args.gene).replace("network_", "").replace(".py", "")
    overrides, config = gene_overrides(args.gene)
    print("gene config:", config, flush=True)
    run_dir = os.path.join(os.environ.get("XQC_RUNS", os.path.join(HERE, "runs")), "llmge_video", name)
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "train_xqc.py"),
           f"env={args.env}", f"seed={args.seed}", f"num_seeds={args.num_seeds}",
           f"max_steps={args.timesteps}", f"eval_interval={min(25_000, args.timesteps)}",
           f"start_training={min(5000, args.timesteps // 2)}", f"run_dir={run_dir}",
           "checkpoint_interval=0", "save_replay_buffer=false", "final_descriptors=true", "resume=never",
           f"final_video={os.path.join(run_dir, 'video')}", f"video_steps={args.video_steps}", *overrides]
    print("Running:", " ".join(cmd), flush=True)
    with open(os.path.join(run_dir, "train.log"), "w") as log:
        rc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"exit {rc}; outputs in {run_dir}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
