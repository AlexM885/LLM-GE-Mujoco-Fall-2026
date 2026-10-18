#!/bin/bash
#SBATCH --job-name=pga_orig
#SBATCH -t 8:00:00
#SBATCH --mem 32G
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --output=slurm-pga-orig-%j.out
# The original PGA-MAP-Elites code, unchanged, on the paper's QDWalker task
# (QDWalker2DBulletEnv-v0, PyBullet) with the paper's configuration.
#
# Usage, from the repository root:
#   sbatch sota/MujocoRL/Baseline/reproduction/slurm/pga_original.sh [SEED] [RUN_DIR]
#
# The paper runs 1,000,000 evaluations; that does not fit in one 8-hour job and
# the original code has no resume. The progress file is written after every
# batch, so whatever is reached is compared with the paper's curves at the same
# number of evaluations. The first job also builds the environment (Python 3.8,
# torch 1.7.1, gym 0.15.4, pybullet 3.0.8 compiled from source, QDgym).
echo "launching original PGA-MAP-Elites"
hostname
module load uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
PGA_DIR="$REPO_ROOT/sota/MujocoRL/Baseline/reproduction/PGA-MAP-Elites-original"
SEED="${1:-0}"
RUN_DIR="${2:-$REPO_ROOT/baseline_runs/reproduction/pga_original/seed${SEED}_job${SLURM_JOB_ID:-$$}}"
mkdir -p "$RUN_DIR"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
# As recommended in the original README
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# main.py reads the cached centroids from CVT/ relative to the working directory
cd "$PGA_DIR"
env -u VIRTUAL_ENV uv run --project "$PGA_DIR" python main.py \
    --config_file configure_experiment/QDWalker_config.txt \
    --seed "$SEED" \
    --save_path "$RUN_DIR"
echo "original PGA-MAP-Elites finished with status $?"
