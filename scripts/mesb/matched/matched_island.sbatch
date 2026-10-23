#!/bin/bash
#SBATCH --job-name=matched-island
#SBATCH -t 16:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=mujoco_rl_output/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# One process of the matched MESB vs fixed-grid experiment on Shazeb's XQC
# island. Same environment and arguments as his run.sh (LLMGE_ALGO=xqc,
# prompt group Mujoco/XQC, llama3), but it runs run_improved_matched.py: his
# run_improved.py plus the hooks in src/mesb_matched.py, generated into the run
# directory by scripts/mesb/matched/build_matched_driver.py.
#
# Required environment (scripts/mesb/matched/launch_pair.sh sets them):
#   PAIR             e.g. xqc_matched_p1
#   ROLE             gen0 | fixed | mesb
# Optional:
#   LLMGE_POP_SIZE (16), MATCHED_GENE_TIME (3:00:00), MATCHED_FIX_CROSSOVER (1)
# Generations: his constants set migration_gen = 5, so an island stops after
# generation 5 (6 generations, 0-5, as in xqc_island_run2) whatever
# LLMGE_NUM_GENERATIONS says.
#
# ROLE=gen0 creates and evaluates generation 0 once and saves it. ROLE=fixed /
# mesb copy that snapshot into their own run directory the first time, then
# resume from it (and from their own checkpoints after that), so both arms
# start from the same 16 evaluated genes.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
REPO="$PWD"

: "${PAIR:?set PAIR}"
: "${ROLE:?set ROLE (gen0, fixed or mesb)}"
BASE="mujoco_rl_output/matched/${PAIR}"
RUN_DIR="${BASE}/${ROLE}"
GEN0_DIR="${BASE}/gen0"
ISLAND="island_llama3_Mujoco-XQC"

if command -v module >/dev/null 2>&1; then
    module load uv 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR" "$RUN_DIR" mujoco_rl_output/slurm_logs
export SERVER_HOSTNAME="$(hostname)"
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export LLMGE_PORT="${LLMGE_PORT:-8169}"

# Shazeb's island settings (xqc_island_run2), the same for every role.
export LLMGE_ALGO=xqc
export LLMGE_POP_SIZE="${LLMGE_POP_SIZE:-16}"
# The two agreed fixes, identical in both arms.
export MATCHED_FIX_CROSSOVER="${MATCHED_FIX_CROSSOVER:-1}"
export MATCHED_GENE_TIME="${MATCHED_GENE_TIME:-3:00:00}"

case "$ROLE" in
    gen0)  export MATCHED_ARCHIVE=fixed MATCHED_STOP_AFTER_GEN0=1 ;;
    fixed) export MATCHED_ARCHIVE=fixed MATCHED_STOP_AFTER_GEN0=0 ;;
    mesb)  export MATCHED_ARCHIVE=sliding MATCHED_STOP_AFTER_GEN0=0 ;;
    *) echo "ROLE must be gen0, fixed or mesb" >&2; exit 2 ;;
esac

if [[ "$ROLE" != gen0 && ! -d "$RUN_DIR/global_data" ]]; then
    for f in "$GEN0_DIR/$ISLAND/checkpoint_gen_0.pkl" "$GEN0_DIR/global_data/global_gen_0.pkl"; do
        [[ -f "$f" ]] || { echo "missing $f: run ROLE=gen0 first" >&2; exit 3; }
    done
    mkdir -p "$RUN_DIR/$ISLAND" "$RUN_DIR/global_data"
    cp "$GEN0_DIR/$ISLAND/checkpoint_gen_0.pkl" "$RUN_DIR/$ISLAND/"
    cp "$GEN0_DIR/global_data/global_gen_0.pkl" "$RUN_DIR/global_data/"
    echo "copied the shared generation 0 from $GEN0_DIR"
fi
# His driver runs each launch to the next multiple of migration_gen (5), so a
# chained part started after generation 5 would run generations 6-10. Stop here.
LAST_GEN="${MATCHED_LAST_GEN:-5}"
if [[ "$ROLE" != gen0 && -f "$RUN_DIR/$ISLAND/checkpoint_gen_${LAST_GEN}.pkl" ]]; then
    echo "$ROLE arm already finished generation $LAST_GEN; nothing to do"
    exit 0
fi
if [[ "$ROLE" == gen0 && -f "$RUN_DIR/$ISLAND/checkpoint_gen_0.pkl" ]]; then
    echo "generation 0 already saved in $RUN_DIR; nothing to do"
    exit 0
fi

DRIVER="$RUN_DIR/run_improved_matched.py"
uv run python scripts/mesb/matched/build_matched_driver.py --out "$DRIVER"
{
    echo "{\"pair\": \"$PAIR\", \"role\": \"$ROLE\", \"archive\": \"$MATCHED_ARCHIVE\","
    echo " \"git_commit\": \"$(git rev-parse HEAD)\", \"run_improved_sha256\": \"$(sha256sum run_improved.py | cut -c1-16)\","
    echo " \"pop_size\": $LLMGE_POP_SIZE, \"generations\": \"0-5 (migration_gen)\","
    echo " \"fix_crossover\": \"$MATCHED_FIX_CROSSOVER\", \"gene_time\": \"$MATCHED_GENE_TIME\","
    echo " \"job\": \"${SLURM_JOB_ID:-local}\", \"started\": \"$(date -Iseconds)\"}"
} >> "$RUN_DIR/matched_runs.jsonl"

echo "matched island: pair=$PAIR role=$ROLE archive=$MATCHED_ARCHIVE pop=$LLMGE_POP_SIZE"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" uv run python "$DRIVER" \
    --checkpoints "$RUN_DIR/$ISLAND" \
    --global_path "$RUN_DIR/global_data" \
    --llm_model llama3 \
    --prompt_group Mujoco/XQC
