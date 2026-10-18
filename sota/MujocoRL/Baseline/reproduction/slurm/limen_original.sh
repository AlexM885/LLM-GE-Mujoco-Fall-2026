#!/bin/bash
#SBATCH --job-name=limen_orig
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 32G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --output=slurm-limen-orig-%j.out
#SBATCH --signal=B:USR1@600
# The original LIMEN code on one of the paper's tasks with the paper's settings:
# 30 iterations and the task's own config. The only code change is a client for
# the LLM-GE model server (limen/llm_client.py, limen/config.py).
#
# Usage, from the repository root:
#   sbatch sota/MujocoRL/Baseline/reproduction/slurm/limen_original.sh TASK [RUN_DIR]
# TASK is one of: easy, medium, hard (XLand-MiniGrid), panda, go1 (MuJoCo Playground).
#
# LLM: by default the LLM-GE server (Llama 3.3 70B; start it with sbatch server.sh).
# LIMEN_LLM=sonnet uses Claude Sonnet 4.6 through OpenRouter, as in the paper; that
# needs OPENAI_API_KEY exported before sbatch or written in LIMEN-original/.env
# The first job builds the environment as the LIMEN README describes.
# Ten minutes before the time limit the job resubmits itself, and the next job
# resumes from the last finished iteration.
echo "launching original LIMEN"
hostname
module load uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
LIMEN_DIR="$REPO_ROOT/sota/MujocoRL/Baseline/reproduction/LIMEN-original"
DEPS_DIR="$REPO_ROOT/baseline_runs/reproduction/_deps"
VENV="$DEPS_DIR/limen_original_venv"
TASK="${1:-easy}"
RUN_DIR="${2:-$REPO_ROOT/baseline_runs/reproduction/limen_original/${TASK}_job${SLURM_JOB_ID:-$$}}"
LLM="${LIMEN_LLM:-llama}"

case "$TASK" in
    easy)   CONFIG=configs/easy_pickup.yaml;          TASK_TEXT="Pick up the blue pyramid." ;;
    medium) CONFIG=configs/medium_place_near.yaml;    TASK_TEXT="Place the yellow pyramid adjacent to the green square." ;;
    hard)   CONFIG=configs/hard_rule_chain.yaml;      TASK_TEXT="Pick up the blue pyramid (transforms to green ball), place near yellow hex." ;;
    panda)  CONFIG=configs/panda_pick_and_track.yaml; TASK_TEXT="" ;;
    go1)    CONFIG=configs/go1_push_recovery.yaml;    TASK_TEXT="" ;;
    *) echo "Unknown task '$TASK' (easy, medium, hard, panda, go1)"; exit 1 ;;
esac

if [ "$LLM" = sonnet ] && [ -z "$OPENAI_API_KEY" ] && [ ! -f "$LIMEN_DIR/.env" ]; then
    echo "No OpenRouter key: export OPENAI_API_KEY or create $LIMEN_DIR/.env"; exit 1
fi
export LLMGE_PORT="${LLMGE_PORT:-8169}"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR" "$DEPS_DIR" "$(dirname "$RUN_DIR")"

# Environment exactly as in the LIMEN README: Python 3.10, the package, JAX with CUDA,
# and MuJoCo Playground for the robot tasks. Built once, then reused.
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.10 "$VENV"
    uv pip install --python "$VENV/bin/python" -e "$LIMEN_DIR"
    # CUDA build of the same JAX version the rest of LIMEN resolved to; upgrading
    # JAX past it would leave jaxlib, flax and brax mismatched
    JAX_VERSION=$("$VENV/bin/python" -c "import jax; print(jax.__version__)")
    uv pip install --python "$VENV/bin/python" "jax[cuda12]==$JAX_VERSION"
fi
if [[ "$TASK" == panda || "$TASK" == go1 ]] && ! "$VENV/bin/python" -c "import mujoco_playground" 2>/dev/null; then
    [ -d "$DEPS_DIR/mujoco_playground" ] || git clone https://github.com/google-deepmind/mujoco_playground.git "$DEPS_DIR/mujoco_playground"
    uv pip install --python "$VENV/bin/python" -e "$DEPS_DIR/mujoco_playground"
fi

# The task's config, unchanged except for the LLM section, saved with the run
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/config.yaml" ] || "$VENV/bin/python" - "$LIMEN_DIR/$CONFIG" "$RUN_DIR/config.yaml" "$LLM" <<'EOF'
import sys, yaml
src, dst, llm = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
if llm == "sonnet":
    cfg["llm"]["model_name"] = "anthropic/claude-sonnet-4.6"
else:
    cfg["llm"].update(provider="llmge", api_base="auto", model_name="meta-llama/Llama-3.3-70B-Instruct",
                      chat_template="llama3", max_tokens=4096, timeout=3600, max_concurrent_requests=8)
# Save after every iteration so that a resubmitted job loses at most one
cfg["checkpoint_interval"] = 1
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("LLM settings:", cfg["llm"])
EOF

# run.py reads .env, contexts/ and rulesets/ relative to the LIMEN folder
cd "$LIMEN_DIR"
ARGS=(--config "$RUN_DIR/config.yaml" --iterations 30 --run-dir "$RUN_DIR")
[ -n "$TASK_TEXT" ] && ARGS+=(--task "$TASK_TEXT")
[ -f "$RUN_DIR/controller_state.json" ] && ARGS+=(--resume "$RUN_DIR") && echo "Resuming from $RUN_DIR"

resubmit() {
    echo "Time limit near; resubmitting to continue in $RUN_DIR"
    (cd "$REPO_ROOT" && sbatch "$REPO_ROOT/sota/MujocoRL/Baseline/reproduction/slurm/limen_original.sh" "$TASK" "$RUN_DIR")
    kill "$PID"
}
trap resubmit USR1
"$VENV/bin/python" run.py "${ARGS[@]}" &
PID=$!
wait "$PID"; STATUS=$?
wait "$PID" 2>/dev/null
echo "original LIMEN finished with status $STATUS"
