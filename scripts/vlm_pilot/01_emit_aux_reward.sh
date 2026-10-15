#!/usr/bin/env bash
# Submit one Gemma 3 27B inference job that emits the aux_reward cache.
# Reads the captures left by 00_capture_parents.sh.
set -euo pipefail
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT first}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

GENE_ID="vlm_pilot_seed"
BEST_SO_FAR=$(python3 - <<PY
import json, os, sys
p = os.path.join("$VLM_PILOT_ROOT", "stats", "${GENE_ID}_stats.json")
try:
    print(json.load(open(p)).get("mean_reward", ""))
except Exception:
    print("")
PY
)

AUX_MODEL_ID="${AUX_MODEL_ID:-meta-llama/Llama-3.2-11B-Vision-Instruct}"

JOB=$(AUX_TOP_GENES="$GENE_ID" \
    AUX_BOTTOM_GENES="" \
    AUX_CACHE_PATH="$VLM_PILOT_ROOT/aux_reward_cache.json" \
    AUX_GENERATION=0 \
    AUX_BEST_SO_FAR="$BEST_SO_FAR" \
    AUX_ARCHIVE_COVERAGE=1 \
    AUX_MODEL_ID="$AUX_MODEL_ID" \
    FEEDBACK_CAPTURE_DIR="$VLM_PILOT_ROOT/captures" \
    MUJOCO_TRAINED_MODEL_DIR="$VLM_PILOT_ROOT/trained_models" \
    sbatch --parsable \
        --job-name=vlm_pilot_aux_emit \
        --output="$VLM_PILOT_ROOT/slurm_aux_emit_%j.out" \
        --export=ALL \
        aux_reward_observer.sh)
echo "{\"aux_emit_job\": $JOB}" > "$VLM_PILOT_ROOT/jobs_01.json"
echo "[emit] aux-reward emit job id: $JOB"
echo "[emit] cache will be at $VLM_PILOT_ROOT/aux_reward_cache.json"
echo "[emit] run 02_train_arm.sbatch for both arms once this completes"
