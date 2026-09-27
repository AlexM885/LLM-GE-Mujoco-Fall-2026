#!/bin/bash
#SBATCH --job-name=chain_vlm_after_pilot
#SBATCH -t 00:30:00
#SBATCH --mem 4G
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/chain-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load uv

CHECKPOINTS="${CHECKPOINTS:-mujoco_islands_run/island_llama3_Mujoco-Normal}"
GLOBAL_PATH="${GLOBAL_PATH:-mujoco_islands_run/global_data}"

mkdir -p run_job_outputs/server run_job_outputs/islands run_job_outputs/evolution \
    run_job_outputs/evaluation run_job_outputs/observer

echo "Preparing visual-feedback continuation from ${CHECKPOINTS}"
GENES="$(uv run python src/feedback_prefetch.py --checkpoints "$CHECKPOINTS" --limit "${CHAIN_PARENT_LIMIT:-4}" | paste -sd, -)"
if [ -z "$(echo "$GENES" | tr -d '[:space:]')" ]; then
    echo "No parent genes found; cannot start observer/VLM continuation." >&2
    exit 1
fi

echo "Parent genes: ${GENES}"
OBSERVER_MODEL_ID="${LLMGE_OBSERVER_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
FEEDBACK_FRAMES="${LLMGE_FEEDBACK_FRAMES:-32}"
OBSERVER_JOB="$(sbatch --parsable \
    --export=ALL,LLMGE_FEEDBACK_MODE=visual,LLMGE_FEEDBACK_FRAMES=${FEEDBACK_FRAMES},LLMGE_OBSERVER_MODEL_ID=${OBSERVER_MODEL_ID},LLMGE_OBSERVER_TORCH_DTYPE=bfloat16 \
    observer.sh "$GENES" visual)"
echo "Submitted observer job ${OBSERVER_JOB}"

VLM_DEPENDENCY="afterok:${OBSERVER_JOB}"
if [ "${CHAIN_REUSE_SERVER:-0}" = "1" ]; then
    echo "Reusing existing LLM server from hostname.log"
else
    rm -f hostname.log
    SERVER_JOB="$(sbatch --parsable --dependency=afterok:${OBSERVER_JOB} server.sh)"
    echo "Submitted continuation server job ${SERVER_JOB}"
fi

VLM_JOB="$(sbatch --parsable --dependency=${VLM_DEPENDENCY} \
    --export=ALL,LLM_SERVER_READY_TIMEOUT=7200,LLM_SERVER_READY_CHECK_INTERVAL=30,LLMGE_NUM_GENERATIONS=3,LLMGE_START_POPULATION_SIZE=4,LLMGE_POPULATION_SIZE=4,LLMGE_NUM_ELITES=2,LLMGE_MIGRATION_GEN=0,LLMGE_CROSSOVER_PROBABILITY=0.0,LLMGE_MUTATION_PROBABILITY=1.0,MUJOCO_EVAL_TIMESTEPS=100000,MUJOCO_EVAL_EPISODES=3,MUJOCO_EVAL_MAX_STEPS=1000,LLMGE_FEEDBACK_MODE=visual,LLMGE_FEEDBACK_REQUIRED=0,LLMGE_FEEDBACK_FRAMES=${FEEDBACK_FRAMES},LLMGE_OBSERVER_MODEL_ID=${OBSERVER_MODEL_ID},LLMGE_OBSERVER_TORCH_DTYPE=bfloat16 \
    run_vlm.sh)"
echo "Submitted visual-feedback continuation job ${VLM_JOB}"
echo "Job Done"
