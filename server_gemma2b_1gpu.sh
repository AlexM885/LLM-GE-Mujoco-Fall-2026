#!/bin/bash
#SBATCH --job-name=LLMGE_Gemma2B
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH -C "H200"
#SBATCH --mem 80G
#SBATCH -c 8
#SBATCH --output=run_job_outputs/server/slurm-%j.out
set -euo pipefail

echo "launching LLM Server"
hostname

module load cuda
module load uv

source scripts/pace_cache_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SERVER_HOSTNAME="$(hostname)"
export LLMGE_PORT="${LLMGE_PORT:-8169}"

HOSTNAME_FILE="$(pwd)/hostname.log"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"
echo "Starting LLM server on port ${LLMGE_PORT}"

uv pip install tiktoken sentencepiece >/dev/null
uv run python -m uvicorn server:app --host "$SERVER_HOSTNAME" --port "$LLMGE_PORT" --workers 1
