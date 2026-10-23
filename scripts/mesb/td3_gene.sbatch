#!/bin/bash
#SBATCH --job-name=mesb-td3-gene
#SBATCH -C nvidia-gpu
#SBATCH -t 8:00:00
#SBATCH -G 1
#SBATCH --mem 16G
#SBATCH -c 12
#SBATCH -N 1
# ---------------------------------------------------------------------------
# One TD3 gene for MESB: train + evaluate on one GPU, write the MESB result
# JSON. Submitted by run_mesb.py --algo td3 --evaluator slurm (one job per
# gene, sbatch --wait); the driver passes --out-json and the other arguments
# of sota/MujocoRL/train_gene.py.
#
# TD3 takes a gradient step per environment step, so 500k steps want a GPU
# and several hours, like run_improved.py's evaluation jobs (8 h limit).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
echo "MESB TD3 gene on $(hostname) | job ${SLURM_JOB_ID:-local}"

if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
    module load uv 2>/dev/null || true
fi
nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU visible; TD3 will run on CPU (slow)"

# The driver passes --device from its own settings (cpu by default); the last
# --device wins, so this job trains on the GPU it was given.
DEVICE=cuda
nvidia-smi -L >/dev/null 2>&1 || DEVICE=cpu
env -u VIRTUAL_ENV uv run --project sota/MujocoRL/eval_env \
    python sota/MujocoRL/train_gene.py "$@" --device "$DEVICE"
