#!/bin/bash
#SBATCH --job-name=walker_seed_backup
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --output=run_job_outputs/evaluation/seed-backup-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load cuda
module load uv

export CUDA_VISIBLE_DEVICES=0
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR" run_job_outputs/evaluation \
    sota/MujocoRL/models sota/MujocoRL/trained_models sota/MujocoRL/stats

cp sota/MujocoRL/network.py sota/MujocoRL/models/network_seed.py

env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env python \
    sota/MujocoRL/train_rl.py \
    -network models.network_seed \
    -env "${MUJOCO_ENV_ID:-Walker2d-v5}" \
    -timesteps "${MUJOCO_EVAL_TIMESTEPS:-500000}" \
    -eval_episodes "${MUJOCO_EVAL_EPISODES:-10}" \
    -eval_max_steps "${MUJOCO_EVAL_MAX_STEPS:-1000}"
