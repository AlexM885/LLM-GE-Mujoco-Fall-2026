#!/bin/bash
# Submit one MuJoCo VLM iteration arm on PACE ICE.
#
# Run from a PACE copy of the repo. The script assumes source checkpoint
# artifacts already exist in SOURCE_RUN and all work stays under scratch.
#
# Example:
#   bash scripts/pace_vlm_iterate.sh planner_top32
#
set -euo pipefail

ARM_NAME="${1:?usage: $0 <arm_name> [source_run] [root]}"
SOURCE_RUN="${2:-/storage/ice1/5/8/nsani3/llmge-vlm-highbudget-20260927-173532}"
ROOT="${3:-/storage/ice1/5/8/nsani3/llmge-vlm-iterate-$(date +%Y%m%d)}"
ARM_DIR="$ROOT/$ARM_NAME"

if [[ "$ARM_DIR" != /storage/ice1/* ]]; then
    echo "Refusing to run outside PACE scratch: $ARM_DIR" >&2
    exit 2
fi

mkdir -p "$ARM_DIR"

copy_artifacts() {
    mkdir -p "$ARM_DIR"/{mujoco_islands_run,sota/MujocoRL,mujoco_rl_output}
    mkdir -p "$ARM_DIR"/run_job_outputs/{observer,islands,evolution,evaluation,server}
    rm -rf \
        "$ARM_DIR/mujoco_islands_run/global_data" \
        "$ARM_DIR/mujoco_islands_run/island_llama3_Mujoco-Normal" \
        "$ARM_DIR/mujoco_rl_output" \
        "$ARM_DIR/sota/MujocoRL/stats" \
        "$ARM_DIR/sota/MujocoRL/trained_models" \
        "$ARM_DIR/sota/MujocoRL/models" \
        "$ARM_DIR/sota/MujocoRL/behavior_captures" \
        "$ARM_DIR/sota/MujocoRL/behavior_cache"
    cp -al "$SOURCE_RUN/mujoco_islands_run/global_data" "$ARM_DIR/mujoco_islands_run/"
    cp -al "$SOURCE_RUN/mujoco_islands_run/island_llama3_Mujoco-Normal" "$ARM_DIR/mujoco_islands_run/"
    cp -al "$SOURCE_RUN/mujoco_rl_output" "$ARM_DIR/"
    cp -al "$SOURCE_RUN/sota/MujocoRL/stats" "$ARM_DIR/sota/MujocoRL/"
    cp -al "$SOURCE_RUN/sota/MujocoRL/trained_models" "$ARM_DIR/sota/MujocoRL/"
    cp -al "$SOURCE_RUN/sota/MujocoRL/models" "$ARM_DIR/sota/MujocoRL/"
    if [[ -d "$SOURCE_RUN/sota/MujocoRL/behavior_captures" ]]; then
        cp -al "$SOURCE_RUN/sota/MujocoRL/behavior_captures" "$ARM_DIR/sota/MujocoRL/"
    fi
    mkdir -p "$ARM_DIR/sota/MujocoRL/behavior_cache"
}

snapshot_baseline() {
    cd "$ARM_DIR"
    python3 - <<'PY'
import json, pathlib
root = pathlib.Path(".")
stats = list((root / "sota/MujocoRL/stats").glob("*_stats.json"))
best = (-10**18, None)
for p in stats:
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    r = d.get("mean_reward", d.get("fitness", d.get("reward")))
    if r is not None and float(r) > best[0]:
        best = (float(r), p.name[:-11])
(root / "baseline_snapshot.json").write_text(json.dumps({
    "arm": root.name,
    "source": "checkpoint before this VLM iteration arm",
    "n_stats": len(stats),
    "best": best,
    "genes": [p.name[:-11] for p in stats],
}, indent=2) + "\n")
print(len(stats), best)
PY
}

arm_defaults() {
    # Shared budget: quick enough for iteration, real enough for Monday.
    export LLMGE_NUM_GENERATIONS="${LLMGE_NUM_GENERATIONS:-7}"
    export LLMGE_START_POPULATION_SIZE="${LLMGE_START_POPULATION_SIZE:-8}"
    export LLMGE_POPULATION_SIZE="${LLMGE_POPULATION_SIZE:-8}"
    export LLMGE_NUM_ELITES="${LLMGE_NUM_ELITES:-4}"
    export LLMGE_MIGRATION_GEN="${LLMGE_MIGRATION_GEN:-0}"
    export LLMGE_CROSSOVER_PROBABILITY="${LLMGE_CROSSOVER_PROBABILITY:-0.0}"
    export LLMGE_MUTATION_PROBABILITY="${LLMGE_MUTATION_PROBABILITY:-1.0}"
    export MUJOCO_EVAL_TIMESTEPS="${MUJOCO_EVAL_TIMESTEPS:-250000}"
    export MUJOCO_EVAL_EPISODES="${MUJOCO_EVAL_EPISODES:-3}"
    export MUJOCO_EVAL_MAX_STEPS="${MUJOCO_EVAL_MAX_STEPS:-1000}"
    export LLM_SERVER_READY_TIMEOUT="${LLM_SERVER_READY_TIMEOUT:-7200}"
    export LLM_SERVER_READY_CHECK_INTERVAL="${LLM_SERVER_READY_CHECK_INTERVAL:-30}"
    export LLMGE_FEEDBACK_REQUIRED="${LLMGE_FEEDBACK_REQUIRED:-1}"
    export LLMGE_FEEDBACK_SELECT_OBSERVED_ONLY="${LLMGE_FEEDBACK_SELECT_OBSERVED_ONLY:-1}"
    export LLMGE_FORCE_OBSERVE="${LLMGE_FORCE_OBSERVE:-1}"
    export LLMGE_FORCE_CAPTURE="${LLMGE_FORCE_CAPTURE:-1}"
    export LLMGE_OBSERVER_MODEL_ID="${LLMGE_OBSERVER_MODEL_ID:-google/gemma-3-27b-it}"
    export LLMGE_OBSERVER_TORCH_DTYPE="${LLMGE_OBSERVER_TORCH_DTYPE:-bfloat16}"
    export LLMGE_OBSERVER_MAX_NEW_TOKENS="${LLMGE_OBSERVER_MAX_NEW_TOKENS:-768}"
    export LLMGE_PARENT_STRATEGY="${LLMGE_PARENT_STRATEGY:-top-fitness}"
    export LLMGE_PARENT_LIMIT="${LLMGE_PARENT_LIMIT:-16}"
    export LLMGE_CACHE_ROOT="${LLMGE_CACHE_ROOT:-$ROOT/.shared_cache_common}"
    export LLMGE_MODEL_PATH="${LLMGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/}"
    export LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-1648}"
    export LLMGE_SERVER_BATCH_SIZE="${LLMGE_SERVER_BATCH_SIZE:-8}"
    export LLMGE_SERVER_BATCH_WAIT_TIME="${LLMGE_SERVER_BATCH_WAIT_TIME:-2}"
}

configure_arm() {
    arm_defaults
    case "$ARM_NAME" in
        planner_top32)
            export LLMGE_FEEDBACK_MODE=visual
            export LLMGE_FEEDBACK_STRUCTURED_PLANNER=1
            export LLMGE_FEEDBACK_FRAMES=32
            export LLMGE_FEEDBACK_FRAME_SAMPLING=bookend
            export LLMGE_FEEDBACK_RENDER_STRIDE=5
            export LLMGE_FEEDBACK_PROMPT_REVISION=v5-planner-top32-observed
            ;;
        planner_top64_uniform)
            export LLMGE_FEEDBACK_MODE=visual
            export LLMGE_FEEDBACK_STRUCTURED_PLANNER=1
            export LLMGE_FEEDBACK_FRAMES=64
            export LLMGE_FEEDBACK_FRAME_SAMPLING=uniform
            export LLMGE_FEEDBACK_RENDER_STRIDE=5
            export LLMGE_FEEDBACK_PROMPT_REVISION=v5-planner-top64-uniform-observed
            ;;
        actionable_top32)
            export LLMGE_FEEDBACK_MODE=visual
            export LLMGE_FEEDBACK_STRUCTURED_PLANNER=0
            export LLMGE_FEEDBACK_FRAMES=32
            export LLMGE_FEEDBACK_FRAME_SAMPLING=bookend
            export LLMGE_FEEDBACK_RENDER_STRIDE=5
            export LLMGE_FEEDBACK_PROMPT_REVISION=v5-actionable-top32-observed
            ;;
        fast_codellama_actionable32)
            export LLMGE_FEEDBACK_MODE=visual
            export LLMGE_FEEDBACK_STRUCTURED_PLANNER=0
            export LLMGE_FEEDBACK_FRAMES=32
            export LLMGE_FEEDBACK_FRAME_SAMPLING=bookend
            export LLMGE_FEEDBACK_RENDER_STRIDE=5
            export LLMGE_FEEDBACK_PROMPT_REVISION=v5-actionable-top32-codellama-observed
            export LLMGE_MODEL_PATH=/storage/ice-shared/vip-vvk/llm_storage/hub/models--codellama--CodeLlama-13b-Instruct-hf
            export LLM_MAX_NEW_TOKENS=768
            export LLMGE_SERVER_BATCH_SIZE=4
            export LLMGE_SERVER_BATCH_WAIT_TIME=1
            ;;
        telemetry_top)
            export LLMGE_FEEDBACK_MODE=telemetry
            export LLMGE_FEEDBACK_STRUCTURED_PLANNER=1
            export LLMGE_FEEDBACK_FRAMES=32
            export LLMGE_FEEDBACK_FRAME_SAMPLING=bookend
            export LLMGE_FEEDBACK_RENDER_STRIDE=5
            export LLMGE_FEEDBACK_PROMPT_REVISION=v5-telemetry-planner-top-observed
            ;;
        *)
            echo "Unknown arm: $ARM_NAME" >&2
            echo "Known: planner_top32 planner_top64_uniform actionable_top32 telemetry_top" >&2
            exit 2
            ;;
    esac
}

submit_jobs() {
    cd "$ARM_DIR"
    module load uv
    GENES=$(uv run python src/feedback_prefetch.py \
        --checkpoints mujoco_islands_run/island_llama3_Mujoco-Normal \
        --strategy "$LLMGE_PARENT_STRATEGY" \
        --limit "$LLMGE_PARENT_LIMIT" | grep -v '^#' | paste -sd, -)
    echo "$GENES" > selected_parent_genes.csv
    echo "Selected parents: $GENES"

    COMMON_EXPORT="ALL"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_CACHE_ROOT=$LLMGE_CACHE_ROOT"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_HF_TOKEN_FILE=$ROOT/.hf_token"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_OBSERVER_MODEL_ID=$LLMGE_OBSERVER_MODEL_ID"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_OBSERVER_TORCH_DTYPE=$LLMGE_OBSERVER_TORCH_DTYPE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_OBSERVER_MAX_NEW_TOKENS=$LLMGE_OBSERVER_MAX_NEW_TOKENS"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_MODE=$LLMGE_FEEDBACK_MODE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_REQUIRED=$LLMGE_FEEDBACK_REQUIRED"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_SELECT_OBSERVED_ONLY=$LLMGE_FEEDBACK_SELECT_OBSERVED_ONLY"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_STRUCTURED_PLANNER=$LLMGE_FEEDBACK_STRUCTURED_PLANNER"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_FRAMES=$LLMGE_FEEDBACK_FRAMES"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_FRAME_SAMPLING=$LLMGE_FEEDBACK_FRAME_SAMPLING"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_RENDER_STRIDE=$LLMGE_FEEDBACK_RENDER_STRIDE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FEEDBACK_PROMPT_REVISION=$LLMGE_FEEDBACK_PROMPT_REVISION"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FORCE_CAPTURE=$LLMGE_FORCE_CAPTURE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_FORCE_OBSERVE=$LLMGE_FORCE_OBSERVE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_NUM_GENERATIONS=$LLMGE_NUM_GENERATIONS"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_START_POPULATION_SIZE=$LLMGE_START_POPULATION_SIZE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_POPULATION_SIZE=$LLMGE_POPULATION_SIZE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_NUM_ELITES=$LLMGE_NUM_ELITES"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_MIGRATION_GEN=$LLMGE_MIGRATION_GEN"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_CROSSOVER_PROBABILITY=$LLMGE_CROSSOVER_PROBABILITY"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_MUTATION_PROBABILITY=$LLMGE_MUTATION_PROBABILITY"
    COMMON_EXPORT="$COMMON_EXPORT,MUJOCO_EVAL_TIMESTEPS=$MUJOCO_EVAL_TIMESTEPS"
    COMMON_EXPORT="$COMMON_EXPORT,MUJOCO_EVAL_EPISODES=$MUJOCO_EVAL_EPISODES"
    COMMON_EXPORT="$COMMON_EXPORT,MUJOCO_EVAL_MAX_STEPS=$MUJOCO_EVAL_MAX_STEPS"
    COMMON_EXPORT="$COMMON_EXPORT,LLM_SERVER_READY_TIMEOUT=$LLM_SERVER_READY_TIMEOUT"
    COMMON_EXPORT="$COMMON_EXPORT,LLM_SERVER_READY_CHECK_INTERVAL=$LLM_SERVER_READY_CHECK_INTERVAL"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_MODEL_PATH=$LLMGE_MODEL_PATH"
    COMMON_EXPORT="$COMMON_EXPORT,LLM_MAX_NEW_TOKENS=$LLM_MAX_NEW_TOKENS"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_SERVER_BATCH_SIZE=$LLMGE_SERVER_BATCH_SIZE"
    COMMON_EXPORT="$COMMON_EXPORT,LLMGE_SERVER_BATCH_WAIT_TIME=$LLMGE_SERVER_BATCH_WAIT_TIME"

    OBS=$(sbatch --parsable -t 06:00:00 --export="$COMMON_EXPORT" observer.sh "$GENES" "$LLMGE_FEEDBACK_MODE")
    # Start the 70B text server only after the Gemma observer releases its GPU.
    # Running both on the same GPU node can leave PyTorch with an inconsistent
    # CUDA view and crash the server during startup.
    SERVER=$(sbatch --parsable -t 08:00:00 --dependency=afterok:$OBS --export="$COMMON_EXPORT" server.sh)
    RUN=$(sbatch --parsable -t 08:00:00 --dependency=afterok:$OBS --export="$COMMON_EXPORT" run_vlm.sh)
    cat > arm_jobs.json <<JSON
{"arm":"$ARM_NAME","server":"$SERVER","observer":"$OBS","run":"$RUN","root":"$ROOT","arm_dir":"$ARM_DIR"}
JSON
    echo "server=$SERVER observer=$OBS run=$RUN"
}

if [[ ! -f "$ROOT/.hf_token" ]]; then
    echo "Missing $ROOT/.hf_token. Create it with chmod 600 before launching Gemma." >&2
    exit 2
fi

configure_arm
if [[ ! -f "$ARM_DIR/baseline_snapshot.json" ]]; then
    # Copy the current working tree into the arm only when the arm is empty.
    rsync -a --delete \
        --exclude .git --exclude .venv --exclude __pycache__ \
        --exclude run_job_outputs --exclude mujoco_islands_run \
        --exclude mujoco_rl_output --exclude 'sota/MujocoRL/stats' \
        --exclude 'sota/MujocoRL/trained_models' --exclude 'sota/MujocoRL/models' \
        --exclude 'sota/MujocoRL/behavior_captures' --exclude 'sota/MujocoRL/behavior_cache' \
        ./ "$ARM_DIR/"
    copy_artifacts
    snapshot_baseline
fi
submit_jobs
