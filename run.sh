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

uv run python run_improved.py \
	--checkpoints mujoco_islands_run/island_llama3_Mujoco-Normal \
	--global_path mujoco_islands_run/global_data \
	--llm_model llama3 \
	--prompt_group Mujoco/Normal
