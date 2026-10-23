#!/bin/bash
# PACE sbatch for the per-generation VLM aux_reward emitter.
#
# Runs Gemma 3 27B on one GPU, reads cached parent rollouts, and writes a
# single aux_reward cache JSON that `train_rl_vlm.py` picks up for every
# child in the generation via the LLMGE_AUX_REWARD_CACHE environment
# variable.
#
# Required env vars from the caller:
#   AUX_TOP_GENES         comma-separated top-performing parent ids
#   AUX_BOTTOM_GENES      comma-separated bottom-performing parent ids
#   AUX_CACHE_PATH        where to write the aux_reward cache JSON
# Optional:
#   AUX_GENERATION, AUX_BEST_SO_FAR, AUX_ARCHIVE_COVERAGE
#   AUX_FITNESS_MAP (path to JSON)
#   AUX_REFLECTION_PATH   previous generation's cache for Eureka reflection
#   AUX_MODEL_ID          default google/gemma-3-27b-it
#   AUX_MAX_FRAMES_PER_PARENT (default 8)
#SBATCH --job-name=aux_reward_observer
#SBATCH -p ice-gpu
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "gpu-h200|gpu-h100|gpu-a100|gpu-l40s|gpu-rtxpro-blackwell"
#SBATCH --mem 96G
#SBATCH -c 12
#SBATCH --output=run_job_outputs/aux_reward/slurm-%j.out
set -euo pipefail

mkdir -p run_job_outputs/aux_reward
hostname
module load cuda
module load uv
nvidia-smi -L || echo "nvidia-smi not available"

# Writable HF cache in scratch; populated with Gemma 3 12B IT weights by
# a one-shot sbatch download (see goal.md). The shared vip-vvk cache
# only has incomplete snapshots for the allowed VLMs, so we point HF at
# a cache we own and keep offline mode on for reproducibility.
export HF_HOME="${HF_HOME:-/storage/ice1/5/8/nsani3/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

: "${AUX_TOP_GENES:?AUX_TOP_GENES is required}"
: "${AUX_CACHE_PATH:?AUX_CACHE_PATH is required}"

AUX_BOTTOM_GENES="${AUX_BOTTOM_GENES:-}"
AUX_MODEL_ID="${AUX_MODEL_ID:-google/gemma-3-12b-it}"
AUX_MAX_FRAMES_PER_PARENT="${AUX_MAX_FRAMES_PER_PARENT:-8}"
AUX_PROMPT_REVISION="${AUX_PROMPT_REVISION:-v1}"

mkdir -p "$(dirname "$AUX_CACHE_PATH")"

CMD=(env -u VIRTUAL_ENV uv run --isolated --project src/observer_env python src/vlm_aux_reward.py
     --top-gene-ids "$AUX_TOP_GENES"
     --bottom-gene-ids "$AUX_BOTTOM_GENES"
     --cache-path "$AUX_CACHE_PATH"
     --model-id "$AUX_MODEL_ID"
     --max-frames-per-parent "$AUX_MAX_FRAMES_PER_PARENT"
     --prompt-revision "$AUX_PROMPT_REVISION")

if [ -n "${AUX_CAPTURE_ROOT:-${FEEDBACK_CAPTURE_DIR:-}}" ]; then
    CMD+=(--capture-root "${AUX_CAPTURE_ROOT:-${FEEDBACK_CAPTURE_DIR}}")
fi

[ -n "${AUX_GENERATION:-}" ] && CMD+=(--generation "$AUX_GENERATION")
[ -n "${AUX_BEST_SO_FAR:-}" ] && CMD+=(--best-so-far "$AUX_BEST_SO_FAR")
[ -n "${AUX_ARCHIVE_COVERAGE:-}" ] && CMD+=(--archive-coverage "$AUX_ARCHIVE_COVERAGE")
[ -n "${AUX_FITNESS_MAP:-}" ] && CMD+=(--fitness-map "$AUX_FITNESS_MAP")
[ -n "${AUX_REFLECTION_PATH:-}" ] && CMD+=(--reflection-path "$AUX_REFLECTION_PATH")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Aux-reward observer job done"
