#!/bin/bash
# One seed of a PPO/TD3 seed-config learning curve (baseline_curve.py).
# Submitted by pace/launch_baseline.sh, which sets partition, GPU, time and output.
#   ALGO GENE SEED TIMESTEPS are passed as positional arguments.
#SBATCH --job-name=mjc_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
set -uo pipefail

ALGO=$1; GENE=$2; SEED=$3; TIMESTEPS=$4
: "${XQC_REPO:?XQC_REPO must be set (use pace/launch_baseline.sh)}"
# Scratch caches (uv, pip, tmp) so nothing is written to the home quota.
# shellcheck disable=SC1091
source "$XQC_REPO/sota/XQC/pace/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$XQC_REPO/sota/MujocoRL"
echo "host=$(hostname) job=$SLURM_JOB_ID algo=$ALGO gene=$GENE seed=$SEED steps=$TIMESTEPS"
nvidia-smi -L 2>/dev/null || echo "no GPU"
env -u VIRTUAL_ENV uv run --frozen --project eval_env python baseline_curve.py \
    -algo "$ALGO" -gene "$GENE" -seed "$SEED" -timesteps "$TIMESTEPS" \
    -eval_every 25000 -eval_episodes 10 \
    -out_dir "$XQC_RUNS/baseline_${ALGO}_halfcheetah_v4/seed${SEED}"
