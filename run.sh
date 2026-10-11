#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
echo "launching LLM Guided Evolution"
hostname
module load uv
module load cuda

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export LLMGE_PORT="${LLMGE_PORT:-8169}"

# Trainer config for this island (ppo | xqc); exported so the LLM and evaluation
# jobs that run_improved.py submits read the same constants. Example:
#   LLMGE_ALGO=xqc LLMGE_RUN=xqc_island_run LLMGE_POP_SIZE=16 sbatch run.sh
export LLMGE_ALGO="${LLMGE_ALGO:-ppo}"
RUN="${LLMGE_RUN:-mujoco_islands_run}"
if [ "$LLMGE_ALGO" = xqc ]; then PROMPT_GROUP=Mujoco/XQC; else PROMPT_GROUP=Mujoco/Normal; fi
echo "algo=$LLMGE_ALGO run=$RUN prompts=$PROMPT_GROUP pop=${LLMGE_POP_SIZE:-32}"

uv run python run_improved.py \
	--checkpoints "$RUN/island_llama3_${PROMPT_GROUP//\//-}" \
	--global_path "$RUN/global_data" \
	--llm_model llama3 \
	--prompt_group "$PROMPT_GROUP"
