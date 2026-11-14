#!/bin/bash
# Launch one tiered-run experiment with the team's fixed settings. Everyone uses
# this script, so every run differs only in its tier and repeat number.
#
#   scripts/tiered_run.sh <tier> <repeat> [--dry-run]
#
#   tier    control | vlm | emitters | sliding
#   repeat  1, 2, 3, ...  (sets the run name <tier>_r<repeat> and the training seed)
#
# --dry-run prints the settings and the jobs it would submit, and submits nothing.
# How to run, checklist and what to report: docs/TIERED_RUNS.md.
#
# Optional (environment):
#   LLMGE_HOSTNAME_FILE=<file>  use a shared LLM server instead of starting one
#   LLMGE_VLM_HF_HOME=<dir>     Hugging Face cache with google/gemma-3-12b-it (vlm tier only)
#   SERVER_JOBS / CONTROLLER_JOBS  length of the job chains (default 21 x 8 h servers, 10 x 18 h controllers)
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

TIER="${1:-}"; REPEAT="${2:-}"; DRY=0
[ "${3:-}" = "--dry-run" ] && DRY=1
if [[ ! "$TIER" =~ ^(control|vlm|emitters|sliding)$ ]] || [[ ! "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    sed -n '2,17p' "$0"; exit 1
fi

# ---------------------------------------------------------------------------
# FIXED SETTINGS: identical for every tier and repeat. Change only by team
# agreement, in a commit, never on the command line.
# ---------------------------------------------------------------------------
export LLMGE_ROBOT=walker2d                    # Walker2d-v5
export LLMGE_SEED=xqc                          # seed learner: XQC
export XQC_EVAL_TIMESTEPS=300000               # steps per gene (Kendall tau, Nov 14: 300k ranks ~ like 475k)
export XQC_EVAL_SEEDS=3                        # training seeds per gene, averaged into its fitness
export MUJOCO_EVAL_EPISODES=10                 # final evaluation episodes
export LLMGE_POP_SIZE=32                       # genes per generation
export LLMGE_MIGRATION_GEN=0                   # one island, no stops at migration boundaries
export LLMGE_PARETO_FRONT_PER_CELL=0           # 32x32 MAP-Elites, one elite per cell
GENERATIONS=10                                 # compute budget, below
export MUJOCO_TOTAL_TIMESTEPS=$(( GENERATIONS * LLMGE_POP_SIZE * XQC_EVAL_TIMESTEPS * XQC_EVAL_SEEDS ))
export LLMGE_MILESTONE_TIMESTEPS=$(( LLMGE_POP_SIZE * XQC_EVAL_TIMESTEPS * XQC_EVAL_SEEDS ))  # ~1 generation
export LLMGE_XQC_EVAL_INTERVAL=50000           # keep learning curves (eval.csv) for Kendall tau checks
export LLMGE_GENES_PER_JOB=1                   # 2 after the smoke test confirms packing on real genes
export LLMGE_GENE_TIME=6:00:00                 # gene job wall time; tighten from smoke-test runtimes
export LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS=$(( 6 * 3600 + 1800 ))  # > gene wall time; counts from job start
export LLMGE_EVAL_MAX_QUEUE_SECONDS=$(( 48 * 3600 ))
export LLMGE_REUSE_DUPLICATES=1                # identical genes reuse a result instead of retraining
BAD_NODES="atl1-1-03-014-16-0,atl1-1-03-011-3-0"  # failed CUDA init in October
# ---------------------------------------------------------------------------

# Tier: the only thing that differs between runs.
export USE_LLM_EMITTERS=0 LLMGE_USE_VLM=0 LLMGE_SLIDING_BOUNDARIES=0
case "$TIER" in
    control)  ;;
    vlm)      export LLMGE_USE_VLM=1 ;;
    emitters) export USE_LLM_EMITTERS=1 ;;
    sliding)  export LLMGE_SLIDING_BOUNDARIES=1 ;;
esac

# Repeat: own run name and training seeds (repeat 1: seeds 0-2, repeat 2: 1000-1002, ...).
RUN="${TIER}_r${REPEAT}"
export LLMGE_TRAIN_SEED=$(( (REPEAT - 1) * 1000 ))
export SEED="$LLMGE_TRAIN_SEED"                # the same seed for the sliding-boundaries driver
export LLMGE_EXCLUDE_NODES="$BAD_NODES"
export GENE_SBATCH_FLAGS="--exclude=$BAD_NODES"
# Each run gets its own server port, so two servers on one node never clash.
export LLMGE_PORT=$(( 8100 + $(printf '%s' "$RUN" | cksum | cut -d' ' -f1) % 800 ))

if [ -n "${LLMGE_HOSTNAME_FILE:-}" ]; then
    SERVERS=0                                  # shared server: start none, wait for it
    [ -r "$LLMGE_HOSTNAME_FILE" ] || echo "WARNING: $LLMGE_HOSTNAME_FILE not readable yet; controllers will wait for it" >&2
    [ -n "${LLMGE_SHARED_PORT:-}" ] && export LLMGE_PORT="$LLMGE_SHARED_PORT"
else
    SERVERS="${SERVER_JOBS:-21}"
    export LLMGE_HOSTNAME_FILE="$PWD/$RUN/hostname.log"  # this run's own server
fi
CONTROLLERS="${CONTROLLER_JOBS:-10}"

if [ "$TIER" = vlm ] && [ -z "${LLMGE_VLM_HF_HOME:-}" ]; then
    echo "vlm tier: set LLMGE_VLM_HF_HOME to a Hugging Face cache with google/gemma-3-12b-it" >&2; exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Uncommitted changes in the repo: every run must use a committed version. Commit or stash first." >&2
    git status --short --untracked-files=no >&2; [ "$DRY" = 1 ] || exit 1
fi
if [ -e "$RUN/chain_jobs.txt" ] && [ "$DRY" = 0 ]; then
    echo "$RUN already launched (see $RUN/chain_jobs.txt). Pick another repeat number." >&2; exit 1
fi

SPEC=$(cat <<EOF
run=$RUN tier=$TIER repeat=$REPEAT commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) user=$USER date=$(date -Is)
$(env | grep -E '^(LLMGE_|XQC_|MUJOCO_|USE_LLM_EMITTERS|GENE_SBATCH_FLAGS|SEED=)' | sort)
servers=$SERVERS controllers=$CONTROLLERS
EOF
)
echo "$SPEC"
if [ "$DRY" = 1 ]; then
    echo; echo "dry run: would run  scripts/llmge_chain.sh $RUN $SERVERS $CONTROLLERS"; exit 0
fi
mkdir -p "$RUN"
echo "$SPEC" > "$RUN/run_spec.txt"
scripts/llmge_chain.sh "$RUN" "$SERVERS" "$CONTROLLERS"
echo
echo "Launched $RUN. Settings: $RUN/run_spec.txt. Per-generation stats will appear in"
echo "  $RUN/island_llama3_Mujoco-XQC/generation_stats.csv (sliding tier: mujoco_rl_output/$RUN/)"
