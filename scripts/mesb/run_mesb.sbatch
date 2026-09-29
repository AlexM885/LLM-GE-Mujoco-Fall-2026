#!/bin/bash
#SBATCH --job-name=mesb
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=mujoco_rl_output/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# One Slurm job runs the whole MESB evolution. Gene evaluations (PPO on
# HalfCheetah, CPU-bound) run in parallel inside the allocation, one per CPU
# (--workers = SLURM_CPUS_PER_TASK). No GPU is needed for PPO+MLP; the LLM
# operators use whatever src/cfg/constants.py selects (Gemini API by default,
# so GEMINI_API_KEY must be set and the node needs outbound HTTPS).
#
# Submit with scripts/mesb/submit_mesb.sh (creates the log dir first), or:
#   mkdir -p mujoco_rl_output/slurm_logs
#   sbatch --export=ALL,RUN_NAME=mesb_s0,SEED=0 scripts/mesb/run_mesb.sbatch
#
# Every setting below can be overridden through the environment.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-mesb_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-0}"
SELECTION_MODE="${SELECTION_MODE:-mesb}"          # nsga2 | mesb-shadow | mesb
DIMS_X="${DIMS_X:-20}"
DIMS_Y="${DIMS_Y:-20}"
REMAP_FREQUENCY="${REMAP_FREQUENCY:-100}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-none}"
PPO_TIMESTEPS="${PPO_TIMESTEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-30}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                      # e.g. "--mesb-remap-at-generation-end"

# Python commands. Defaults use the two uv projects already in the repo:
# the root project (LLM operators, DEAP) and sota/MujocoRL/eval_env (MuJoCo, SB3).
DRIVER_PYTHON="${DRIVER_PYTHON:-uv run python}"
LLM_PYTHON="${LLM_PYTHON:-uv run python}"
EVAL_PYTHON="${EVAL_PYTHON:-env -u VIRTUAL_ENV uv run --project sota/MujocoRL/eval_env python}"

# Optional site-specific environment setup (modules, conda, tokens).
if [[ -n "${MESB_ENV_SETUP:-}" ]]; then
    eval "$MESB_ENV_SETUP"
elif command -v module >/dev/null 2>&1; then
    module load uv 2>/dev/null || true
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

mkdir -p mujoco_rl_output/slurm_logs
RESUME=""
if [[ -f "mujoco_rl_output/${RUN_NAME}/config.json" ]]; then
    RESUME="--resume"
    echo "Run directory exists: resuming ${RUN_NAME}"
fi

echo "MESB run ${RUN_NAME} on $(hostname) | mode ${SELECTION_MODE} | seed ${SEED} | dims ${DIMS_X}x${DIMS_Y} | remap ${REMAP_FREQUENCY} | workers ${WORKERS}"
# shellcheck disable=SC2086
$DRIVER_PYTHON run_mesb.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --selection-mode "$SELECTION_MODE" \
    --mesb-dims "$DIMS_X" "$DIMS_Y" \
    --mesb-remap-frequency "$REMAP_FREQUENCY" \
    --mesb-buffer-capacity "$BUFFER_CAPACITY" \
    --eval-timesteps "$PPO_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --eval-max-steps "$EVAL_MAX_STEPS" \
    --num-generations "$NUM_GENERATIONS" \
    --workers "$WORKERS" \
    --llm-python "$LLM_PYTHON" \
    --eval-python "$EVAL_PYTHON" \
    $RESUME $EXTRA_ARGS

$DRIVER_PYTHON -c "import matplotlib" 2>/dev/null && ANALYZE="$DRIVER_PYTHON" || ANALYZE="uv run --with matplotlib python"
$ANALYZE sota/MujocoRL/analyze_mesb.py --run-dir "mujoco_rl_output/${RUN_NAME}" || \
    echo "analysis failed; rerun analyze_mesb.py manually"
