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
# (--workers = SLURM_CPUS_PER_TASK). No GPU is needed for PPO+MLP. The LLM
# operators use the team's local LLM server (LOCAL_LLM in constants_Mujoco.py):
# start it first with scripts/mesb/llm_server.sbatch; the operators wait for it
# through hostname.log. (Not needed with EXTRA_ARGS="--llm-backend mock".)
#
# Learner: ALGO=ppo (default; SB3 PPO on this job's CPUs) or ALGO=xqc
# (Shazeb's XQC island: one GPU job per gene via scripts/mesb/xqc_gene.sbatch,
# GPU_JOBS at a time; this driver job then only needs a few CPUs).
#
# For runs longer than 16 h: CHAIN=N scripts/mesb/submit_mesb.sh ... queues N
# jobs back to back, each resuming from the last checkpoint.
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
ALGO="${ALGO:-ppo}"                                # ppo | xqc
SEED="${SEED:-0}"
SELECTION_MODE="${SELECTION_MODE:-mesb}"          # nsga2 | mesb-shadow | mesb
DIMS_X="${DIMS_X:-20}"
DIMS_Y="${DIMS_Y:-20}"
REMAP_FREQUENCY="${REMAP_FREQUENCY:-100}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-none}"
# Training steps per gene; empty = the learner's default (ppo 500000, xqc 100000).
TIMESTEPS="${TIMESTEPS:-${PPO_TIMESTEPS:-}}"
NUM_SEEDS="${NUM_SEEDS:-}"                         # xqc: seeds per gene (default 3)
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
if [[ "$ALGO" == xqc ]]; then
    NUM_GENERATIONS="${NUM_GENERATIONS:-15}"
    WORKERS="${WORKERS:-${GPU_JOBS:-16}}"         # concurrent per-gene GPU jobs
else
    NUM_GENERATIONS="${NUM_GENERATIONS:-30}"
    WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
fi
LLM_WORKERS="${LLM_WORKERS:-8}"                  # concurrent LLM calls (server batches them)
export LLM_SERVER_READY_TIMEOUT="${LLM_SERVER_READY_TIMEOUT:-14400}"  # wait for a restarted server
EXTRA_ARGS="${EXTRA_ARGS:-}"                      # e.g. "--mesb-remap-at-generation-end"

# Python commands. Defaults use the two uv projects already in the repo:
# the root project (driver, LLM operators, DEAP, matplotlib) and
# sota/MujocoRL/eval_env (MuJoCo, SB3).
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

ALGO_ARGS=(--algo "$ALGO")
[[ -n "$TIMESTEPS" ]] && ALGO_ARGS+=(--eval-timesteps "$TIMESTEPS")
[[ -n "$NUM_SEEDS" ]] && ALGO_ARGS+=(--eval-num-seeds "$NUM_SEEDS")
if [[ "$ALGO" == xqc ]]; then
    ALGO_ARGS+=(--evaluator slurm)
    export MESB_GENE_SBATCH_FLAGS="${GENE_SBATCH_FLAGS:-}"
fi

echo "MESB run ${RUN_NAME} on $(hostname) | algo ${ALGO} | mode ${SELECTION_MODE} | seed ${SEED} | dims ${DIMS_X}x${DIMS_Y} | remap ${REMAP_FREQUENCY} | workers ${WORKERS}"
# shellcheck disable=SC2086
$DRIVER_PYTHON run_mesb.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --selection-mode "$SELECTION_MODE" \
    --mesb-dims "$DIMS_X" "$DIMS_Y" \
    --mesb-remap-frequency "$REMAP_FREQUENCY" \
    --mesb-buffer-capacity "$BUFFER_CAPACITY" \
    "${ALGO_ARGS[@]}" \
    --eval-episodes "$EVAL_EPISODES" \
    --eval-max-steps "$EVAL_MAX_STEPS" \
    --num-generations "$NUM_GENERATIONS" \
    --workers "$WORKERS" \
    --llm-workers "$LLM_WORKERS" \
    --llm-python "$LLM_PYTHON" \
    --eval-python "$EVAL_PYTHON" \
    $RESUME $EXTRA_ARGS

$DRIVER_PYTHON sota/MujocoRL/analyze_mesb.py --run-dir "mujoco_rl_output/${RUN_NAME}" || \
    echo "analysis failed; rerun analyze_mesb.py manually"
