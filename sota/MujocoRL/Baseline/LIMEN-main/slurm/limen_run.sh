#!/bin/bash
#SBATCH --job-name=limen_ctrl
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=slurm-limen-%j.out
# LIMEN controller for the Walker2d-v5 / TD3 baseline. The counterpart of run.sh.
#
# Usage, from the repository root, with the LLM server running (sbatch server.sh):
#   sbatch sota/MujocoRL/Baseline/LIMEN-main/slurm/limen_run.sh [RUN_DIR]
#
# The controller only asks the LLM for candidates and waits; every training is
# its own job (slurm/train_job.sh). It stops before the time limit and resubmits
# itself with --resume until the training budget is spent.
echo "launching LIMEN"
hostname
module load uv
module load cuda

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
LIMEN_DIR="$REPO_ROOT/sota/MujocoRL/Baseline/LIMEN-main"
# Everything this run writes goes under RUN_DIR; pass a new RUN_DIR for a new run.
RUN_DIR="${1:-$REPO_ROOT/baseline_runs/limen/walker2d}"
LOG_DIR="$RUN_DIR/job_logs"
mkdir -p "$RUN_DIR/train_tmp" "$LOG_DIR"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export LLMGE_PORT="${LLMGE_PORT:-8169}"
# The controller itself needs no GPU
export JAX_PLATFORMS=cpu

RESUME=()
if [ -f "$RUN_DIR/controller_state.json" ]; then
    echo "Resuming from $RUN_DIR"
    RESUME=(--resume "$RUN_DIR")
fi

uv run --project "$LIMEN_DIR" python "$LIMEN_DIR/run.py" \
    --config "$LIMEN_DIR/configs/gym_walker2d.yaml" \
    --run-dir "$RUN_DIR" "${RESUME[@]}" \
    --launcher "sbatch --wait -o $LOG_DIR/train-%j.out $LIMEN_DIR/slurm/train_job.sh" \
    --work-dir "$RUN_DIR/train_tmp" \
    --max-hours 7.5
STATUS=$?

if [ "$STATUS" -eq 3 ]; then
    echo "Time limit reached before the budget; resubmitting"
    sbatch "$LIMEN_DIR/slurm/limen_run.sh" "$RUN_DIR"
else
    echo "LIMEN finished with status $STATUS"
fi
