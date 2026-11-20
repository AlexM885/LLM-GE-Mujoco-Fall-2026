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
#   CONTROLLER_TIME             wall time of each controller job (default 18:00:00)
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
export LLMGE_XQC_EVAL_INTERVAL=25000           # learning curve (eval.csv) every 25k, same as midterm runs, for Kendall tau
# Gene jobs. PACE caps each user at 32 GPU-hours of *requested* time over running jobs
# (the LLM server holds 16), so the wall time we ask for decides how many genes run at once.
# Timing test (Nov 14, H200): the seed gene trains 300k x 3 seeds in ~24 min alone, and two
# packed on one GPU take ~27 min together. 3 h leaves room for slower GPUs and big evolved genes.
export LLMGE_GENES_PER_JOB=8                   # pack8-h200: eight genes share one H200 job (the GPU idles with 2)
export LLMGE_GENE_TIME=2:30:00                 # cost-cap: ~0.7 h per 1x cost at 300k packed (500k run, Nov 15) -> <= ~1.4 h at 2x; 2.5 h fits 6 jobs in 16 GPU-h
export LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS=$(( 3 * 3600 ))  # > gene wall time; counts from job start
export LLMGE_EVAL_MAX_QUEUE_SECONDS=$(( 48 * 3600 ))
export LLMGE_REUSE_DUPLICATES=1                # identical genes reuse a result instead of retraining
export LLMGE_XQC_COST_CAP=2                    # cost-cap: genes over 2x the seed's training cost are shrunk before training (train_xqc_gene.py)
BAD_NODES="atl1-1-03-014-16-0,atl1-1-03-011-3-0,atl1-1-03-010-25-0"  # fail JAX CUDA init (Oct; 010-25-0 on Nov 14)
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
RUN="p8_${TIER}_r${REPEAT}"                   # pack8-h200: own run names and server port
export LLMGE_TRAIN_SEED=$(( (REPEAT - 1) * 1000 ))
export SEED="$LLMGE_TRAIN_SEED"                # the same seed for the sliding-boundaries driver
export LLMGE_EXCLUDE_NODES="$BAD_NODES"
# Same wall time for the sliding tier's gene jobs (its job script defaults to 1:30:00).
export GENE_SBATCH_FLAGS="-t $LLMGE_GENE_TIME -C H200 -c 24 --mem 64G --exclude=$BAD_NODES"   # pack8-h200: H200 only (V100s were 3-4x slower), cores and RAM for 8 genes
# Sliding tier (run_mesb.py): its gene timeout covers queue + training, so give it the same
# queue limit plus gene time as the other tiers, and keep a whole generation in flight.
export EXTRA_ARGS="--eval-timeout $(( LLMGE_EVAL_MAX_QUEUE_SECONDS + LLMGE_EVAL_NO_PROGRESS_TIMEOUT_SECONDS ))"
export GPU_JOBS="$LLMGE_POP_SIZE"
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
CTIME="${CONTROLLER_TIME:-18:00:00}"

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
$(env | grep -E '^(LLMGE_|XQC_|MUJOCO_|USE_LLM_EMITTERS|GENE_SBATCH_FLAGS|SEED=|EXTRA_ARGS=|GPU_JOBS=)' | sort)
servers=$SERVERS controllers=$CONTROLLERS controller_time=$CTIME
EOF
)
echo "$SPEC"
if [ "$DRY" = 1 ]; then
    echo; echo "dry run: would run  scripts/llmge_chain.sh $RUN $SERVERS $CONTROLLERS $CTIME"; exit 0
fi
mkdir -p "$RUN"
echo "$SPEC" > "$RUN/run_spec.txt"
scripts/llmge_chain.sh "$RUN" "$SERVERS" "$CONTROLLERS" "$CTIME"
echo
echo "Launched $RUN. Settings: $RUN/run_spec.txt. Per-generation stats will appear in"
echo "  $RUN/island_llama3_Mujoco-XQC/generation_stats.csv (sliding tier: mujoco_rl_output/$RUN/)"
