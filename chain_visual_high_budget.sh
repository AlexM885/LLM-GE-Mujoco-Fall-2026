#!/bin/bash
#SBATCH --job-name=chain_vlm_high_budget
#SBATCH -t 01:00:00
#SBATCH --mem 4G
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/high-chain-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load uv

CHECKPOINTS="${CHECKPOINTS:-mujoco_islands_run/island_llama3_Mujoco-Normal}"

mkdir -p run_job_outputs/server run_job_outputs/islands run_job_outputs/evolution \
    run_job_outputs/evaluation run_job_outputs/observer

echo "Preparing high-budget visual-feedback continuation from ${CHECKPOINTS}"
GENES="$(uv run python src/feedback_prefetch.py --checkpoints "$CHECKPOINTS" --limit "${CHAIN_PARENT_LIMIT:-0}" | paste -sd, -)"
if [ -z "$(echo "$GENES" | tr -d '[:space:]')" ]; then
    echo "No parent genes found; cannot start observer/VLM continuation." >&2
    exit 1
fi

echo "Parent genes: ${GENES}"
OBSERVER_MODEL_ID="${LLMGE_OBSERVER_MODEL_ID:-google/gemma-3-27b-it}"
FEEDBACK_FRAMES="${LLMGE_FEEDBACK_FRAMES:-32}"

OBSERVER_JOB="$(sbatch --parsable -t "${OBSERVER_TIME_LIMIT:-04:00:00}" \
    --export=ALL,LLMGE_FEEDBACK_MODE=visual,LLMGE_FEEDBACK_FRAMES=${FEEDBACK_FRAMES},LLMGE_CAPTURE_TIMEOUT_SECONDS=${LLMGE_CAPTURE_TIMEOUT_SECONDS:-600},LLMGE_OBSERVER_MODEL_ID=${OBSERVER_MODEL_ID},LLMGE_OBSERVER_TORCH_DTYPE=bfloat16 \
    observer.sh "$GENES" visual)"
echo "Submitted observer job ${OBSERVER_JOB}"

VLM_JOB="$(sbatch --parsable -t "${VLM_TIME_LIMIT:-24:00:00}" --dependency=afterok:${OBSERVER_JOB} \
    --export=ALL,LLM_SERVER_READY_TIMEOUT=7200,LLM_SERVER_READY_CHECK_INTERVAL=30,LLMGE_NUM_GENERATIONS=${LLMGE_NUM_GENERATIONS:-6},LLMGE_START_POPULATION_SIZE=${LLMGE_START_POPULATION_SIZE:-32},LLMGE_POPULATION_SIZE=${LLMGE_POPULATION_SIZE:-32},LLMGE_NUM_ELITES=${LLMGE_NUM_ELITES:-8},LLMGE_MIGRATION_GEN=0,LLMGE_CROSSOVER_PROBABILITY=${LLMGE_CROSSOVER_PROBABILITY:-0.0},LLMGE_MUTATION_PROBABILITY=${LLMGE_MUTATION_PROBABILITY:-1.0},MUJOCO_EVAL_TIMESTEPS=${MUJOCO_EVAL_TIMESTEPS:-500000},MUJOCO_EVAL_EPISODES=${MUJOCO_EVAL_EPISODES:-3},MUJOCO_EVAL_MAX_STEPS=${MUJOCO_EVAL_MAX_STEPS:-1000},LLMGE_FEEDBACK_MODE=visual,LLMGE_FEEDBACK_REQUIRED=0,LLMGE_FEEDBACK_FRAMES=${FEEDBACK_FRAMES},LLMGE_OBSERVER_MODEL_ID=${OBSERVER_MODEL_ID},LLMGE_OBSERVER_TORCH_DTYPE=bfloat16 \
    run_vlm.sh)"
echo "Submitted visual-feedback continuation job ${VLM_JOB}"
echo "Job Done"
