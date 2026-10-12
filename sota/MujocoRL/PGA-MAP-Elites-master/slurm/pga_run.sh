#!/bin/bash
#SBATCH --job-name=pga_me
#SBATCH -t 8:00:00
#SBATCH --mem 32G
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --output=slurm-pga-%j.out
# PGA-MAP-Elites baseline on Walker2d-v5. One CPU-only job, no LLM server.
#
# Usage, from the repository root:
#   sbatch sota/MujocoRL/PGA-MAP-Elites-master/slurm/pga_run.sh [SEED] [RUN_DIR]
#
# Walker2d_config.txt starts 24 evaluation and 8 variation processes; keep
# -c in line with --num_cpu / --num_cpu_var if you change them.
echo "launching PGA-MAP-Elites"
hostname
module load uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PGA_DIR="$REPO_ROOT/sota/MujocoRL/PGA-MAP-Elites-master"
SEED="${1:-0}"
# Everything this run writes goes under RUN_DIR; the job id keeps reruns apart.
RUN_DIR="${2:-$REPO_ROOT/baseline_runs/pga_map_elites/seed${SEED}_job${SLURM_JOB_ID:-$$}}"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# The search stops after --max_hours even if the step budget is not spent, so
# that the archive, the models and the final evaluation are written before the
# job's time limit. There is no resume.
env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env python "$PGA_DIR/main.py" \
    --config_file "$PGA_DIR/configure_experiment/Walker2d_config.txt" \
    --seed "$SEED" \
    --save_path "$RUN_DIR" \
    --max_hours 6.5
echo "PGA-MAP-Elites finished with status $?"
