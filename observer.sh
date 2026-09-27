#!/bin/bash
#SBATCH --job-name=observe_parents
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "H200"
#SBATCH --mem 80G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --output=run_job_outputs/observer/slurm-%j.out
#
# Populate the sparse-visual-feedback cache for selected trained parents.
#
# Usage (after at least one generation has trained checkpoints):
#   GENES=$(uv run python src/feedback_prefetch.py --checkpoints <ckpt_dir>)
#   sbatch observer.sh "$GENES" visual
#
# Arguments:
#   $1  comma-separated gene ids, or a path to a file with one id per line
#   $2  mode: "visual" (frames + telemetry) or "telemetry" (text control)
#
# The job renders frames in the isolated MuJoCo eval environment, then loads the
# frozen observer model once and processes every parent in a single batch.
set -euo pipefail

module load cuda
module load uv

export CUDA_VISIBLE_DEVICES=0
# Headless offscreen rendering for frame capture.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LLMGE_USE_SCRATCH_HF=1
source scripts/pace_cache_env.sh
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

GENES_INPUT="${1:?usage: observer.sh <gene_ids_csv|gene_file> [visual|telemetry]}"
MODE="${2:-${LLMGE_FEEDBACK_MODE:-visual}}"
BACKEND="${LLMGE_OBSERVER_BACKEND:-hf}"

if [ -f "$GENES_INPUT" ]; then
    GENES="$(paste -sd, "$GENES_INPUT")"
else
    GENES="$GENES_INPUT"
fi
GENES="$(echo "$GENES" | tr -d ' ')"

CAPTURE_ROOT="${LLMGE_FEEDBACK_CAPTURE_DIR:-sota/MujocoRL/behavior_captures}"
EVAL_PROJECT="${LLMGE_MUJOCO_EVAL_PROJECT:-sota/MujocoRL/eval_env}"
OBSERVER_PROJECT="${LLMGE_OBSERVER_PROJECT:-src/observer_env}"

export LLMGE_FEEDBACK_MODE="$MODE"
export LLMGE_FEEDBACK_CAPTURE_DIR="$CAPTURE_ROOT"
export LLMGE_FEEDBACK_FRAMES="${LLMGE_FEEDBACK_FRAMES:-32}"
export LLMGE_OBSERVER_MODEL_ID="${LLMGE_OBSERVER_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
export LLMGE_OBSERVER_TORCH_DTYPE="${LLMGE_OBSERVER_TORCH_DTYPE:-bfloat16}"
CAPTURE_TIMEOUT_SECONDS="${LLMGE_CAPTURE_TIMEOUT_SECONDS:-900}"

echo "Observer batch job"
echo "  mode=$MODE backend=$BACKEND"
echo "  genes=$GENES"
echo "  capture_root=$CAPTURE_ROOT"
echo "  frames=$LLMGE_FEEDBACK_FRAMES"
echo "  observer_model=$LLMGE_OBSERVER_MODEL_ID"
echo "  capture_timeout_seconds=$CAPTURE_TIMEOUT_SECONDS"

# --- Stage 1: capture ordered frames + telemetry (MuJoCo eval environment) ---
for GENE in $(echo "$GENES" | tr ',' ' '); do
    MODEL="sota/MujocoRL/trained_models/${GENE}.zip"
    if [ ! -f "$MODEL" ]; then
        MODEL="sota/MujocoRL/trained_models/network_${GENE}.zip"
    fi
    GENOME="sota/MujocoRL/models/network_${GENE}.py"
    OUT_DIR="${CAPTURE_ROOT}/${GENE}"
    if [ -f "${OUT_DIR}/manifest.json" ] && [ "${LLMGE_FORCE_CAPTURE:-0}" != "1" ]; then
        echo "  [capture] skip ${GENE} (cached capture)"
        continue
    fi
    if [ ! -f "$MODEL" ]; then
        echo "  [capture] skip ${GENE} (missing checkpoint ${MODEL})"
        continue
    fi
    echo "  [capture] ${GENE}"
    timeout "$CAPTURE_TIMEOUT_SECONDS" env -u VIRTUAL_ENV uv run --isolated --project "$EVAL_PROJECT" python \
        sota/MujocoRL/capture_rollout.py \
        --model "$MODEL" \
        --genome "$GENOME" \
        --gene-id "$GENE" \
        --out-dir "$OUT_DIR" \
        --env "${MUJOCO_ENV_ID:-Walker2d-v5}" \
        --episodes "${LLMGE_FEEDBACK_EPISODES:-2}" \
        --frames "$LLMGE_FEEDBACK_FRAMES" \
        --max-steps "${LLMGE_FEEDBACK_MAX_STEPS:-1000}" \
        --camera "${LLMGE_FEEDBACK_CAMERA:-track}" \
        --image-max-side "${LLMGE_FEEDBACK_IMAGE_MAX_SIDE:-336}" \
        --seed-base "${LLMGE_FEEDBACK_SEED_BASE:-1000}" \
        || echo "  [capture] FAILED or timed out for ${GENE}; continuing"
done

# --- Stage 2: one model load, then observe every parent (observer environment) ---
FORCE_FLAG=""
if [ "${LLMGE_FORCE_OBSERVE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

echo "  [observe] backend=${BACKEND} mode=${MODE}"
env -u VIRTUAL_ENV uv run --isolated --project "$OBSERVER_PROJECT" python \
    src/behavior_observer.py \
    --gene-ids "$GENES" \
    --mode "$MODE" \
    --backend "$BACKEND" \
    ${FORCE_FLAG}

echo "Feedback cache is ready. Run evolution with LLMGE_FEEDBACK_MODE=${MODE}."
echo "Job Done"
