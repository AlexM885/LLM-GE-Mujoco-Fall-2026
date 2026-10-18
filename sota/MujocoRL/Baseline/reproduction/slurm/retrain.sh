#!/bin/bash
#SBATCH --job-name=retrain
#SBATCH -t 1:30:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=slurm-retrain-%A_%a.out
# Train one network file several times, the same way LLM-GE trains a gene
# (sota/MujocoRL/train_rl.py: TD3, 500k steps, 10 evaluation episodes).
# One array task per repeat; each repeat gets a fresh random seed.
#
# Usage, from the repository root:
#   sbatch --array=1-5 sota/MujocoRL/Baseline/reproduction/slurm/retrain.sh NETWORK_FILE NAME
# For example, LLM-GE's best gene and the unchanged seed network:
#   sbatch --array=1-5 sota/MujocoRL/Baseline/reproduction/slurm/retrain.sh sota/MujocoRL/models/network_xXxbGMMPUTN0Xm70dnUhW8Zl7ch.py best_gene
#   sbatch --array=1-5 sota/MujocoRL/Baseline/reproduction/slurm/retrain.sh sota/MujocoRL/network.py seed_network
echo "Launching retrain"
hostname
module load cuda
module load uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
NETWORK_FILE="$1"
NAME="$2"
REPEAT="${SLURM_ARRAY_TASK_ID:-1}"
OUT_DIR="$REPO_ROOT/baseline_runs/reproduction/retrain/$NAME"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"

env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env python \
    sota/MujocoRL/Baseline/reproduction/retrain.py "$NETWORK_FILE" "$OUT_DIR" "${NAME}_r${REPEAT}"
