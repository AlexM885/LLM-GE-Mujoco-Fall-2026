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

source scripts/pace_cache_env.sh

export SERVER_HOSTNAME=$(hostname)
export LLMGE_PORT="${LLMGE_PORT:-8169}"

uv run python run_improved.py \
	--checkpoints mujoco_islands_run/island_llama3_Mujoco-Normal \
	--global_path mujoco_islands_run/global_data \
	--llm_model llama3 \
	--prompt_group Mujoco/Normal
