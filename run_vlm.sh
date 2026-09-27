#!/bin/bash
#SBATCH --job-name=llm_opt_vlm
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
#
# Evolution run with sparse visual feedback enabled.
#
# Workflow on PACE ICE:
#   1. Start the text LLM server:        sbatch server.sh
#   2. Train at least one generation (run.sh) so parents have checkpoints.
#   3. Populate the feedback cache:      sbatch observer.sh "$GENES" visual
#   4. Start this run:                   sbatch run_vlm.sh
#
# The controller only reads the cache; it never needs a GPU or the VLM. Set
# LLMGE_FEEDBACK_MODE=telemetry for the strong text-control condition.
set -euo pipefail

echo "launching LLM Guided Evolution with sparse feedback"
hostname
module load uv
module load cuda

source scripts/pace_cache_env.sh

export SERVER_HOSTNAME="${SERVER_HOSTNAME:-$(hostname)}"
export LLMGE_PORT="${LLMGE_PORT:-8169}"

# Sparse feedback configuration (must match the observer job that filled the cache)
export LLMGE_FEEDBACK_MODE="${LLMGE_FEEDBACK_MODE:-visual}"
export LLMGE_FEEDBACK_REQUIRED="${LLMGE_FEEDBACK_REQUIRED:-0}"
export LLMGE_FEEDBACK_CACHE_DIR="${LLMGE_FEEDBACK_CACHE_DIR:-sota/MujocoRL/behavior_cache}"
export LLMGE_FEEDBACK_FRAMES="${LLMGE_FEEDBACK_FRAMES:-32}"
export LLMGE_OBSERVER_MODEL_ID="${LLMGE_OBSERVER_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
export LLMGE_OBSERVER_TORCH_DTYPE="${LLMGE_OBSERVER_TORCH_DTYPE:-bfloat16}"

echo "Sparse feedback cache settings:"
echo "  mode=$LLMGE_FEEDBACK_MODE"
echo "  frames=$LLMGE_FEEDBACK_FRAMES"
echo "  observer_model=$LLMGE_OBSERVER_MODEL_ID"

uv run python run_improved.py \
	--checkpoints mujoco_islands_run/island_llama3_Mujoco-Normal \
	--global_path mujoco_islands_run/global_data \
	--llm_model llama3 \
	--prompt_group Mujoco/Normal
