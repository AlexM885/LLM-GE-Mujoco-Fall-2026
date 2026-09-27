#!/bin/bash
#SBATCH --job-name=vlm_observe_gene
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "H200"
#SBATCH --mem 80G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --output=run_job_outputs/observer/vlm-gene-%j.out

set -euo pipefail

GENE="${1:?usage: vlm_observe_gene.sh <gene_id> [visual|telemetry]}"
MODE="${2:-visual}"

cd "$SLURM_SUBMIT_DIR"
module load cuda
module load uv

mkdir -p run_job_outputs/observer
export LLMGE_FEEDBACK_MODE="$MODE"
export LLMGE_OBSERVER_BACKEND="${LLMGE_OBSERVER_BACKEND:-hf}"
export LLMGE_FORCE_CAPTURE="${LLMGE_FORCE_CAPTURE:-0}"
export LLMGE_FORCE_OBSERVE="${LLMGE_FORCE_OBSERVE:-0}"

bash observer.sh "$GENE" "$MODE"
