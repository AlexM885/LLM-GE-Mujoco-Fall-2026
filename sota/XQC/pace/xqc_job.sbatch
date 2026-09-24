#!/bin/bash
# One link of an XQC job chain. Submitted by pace/launch_chain.sh, which sets
# --output, --time, partition/QoS/GPU and exports XQC_RUN_DIR, XQC_BASE, XQC_REPO.
#
#SBATCH --job-name=xqc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#
# Behaviour:
#   - resumes from the newest checkpoint in XQC_RUN_DIR (train_xqc.py resume=auto)
#   - Slurm sends USR1 to this script 300 s before the wall limit; it is forwarded
#     to Python, which checkpoints and exits 0. max_runtime_s is also set from the
#     real remaining time as a fallback in case the signal is lost.
#   - when DONE appears, the remaining queued jobs of the chain are cancelled
#   - two consecutive jobs that fail without advancing the checkpoint abort the
#     chain (cancel the rest) instead of burning allocations in a crash loop
set -uo pipefail

: "${XQC_RUN_DIR:?XQC_RUN_DIR must be set (use pace/launch_chain.sh)}"
# shellcheck disable=SC1091
source "${XQC_REPO:?}/sota/XQC/pace/env.sh"

CHAIN_DIR="$XQC_RUN_DIR/chain"
mkdir -p "$CHAIN_DIR"
log() { echo "[chain $(date '+%F %T')] $*"; echo "$(date '+%F %T') job=$SLURM_JOB_ID $*" >> "$CHAIN_DIR/history.log"; }

cancel_rest() {
    # Cancel every job of this chain that is still pending, other than ourselves.
    local ids
    ids=$(grep -v "^${SLURM_JOB_ID}\$" "$CHAIN_DIR/jobs.txt" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    [ -n "$ids" ] && scancel --state=PENDING "${ids//,/ }" 2>/dev/null
    log "cancelled pending chain jobs: ${ids:-none} ($1)"
}

ckpt_step() {
    local p="$XQC_RUN_DIR/checkpoints/latest"
    [ -f "$p" ] && sed -E 's/step_0*([0-9]+)/\1/' "$p" | tr -d '\n' || echo 0
}

remaining_seconds() {
    # squeue %L: time left as [D-]HH:MM:SS, MM:SS or similar
    local left d=0 h=0 m=0 s=0
    left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | tr -d ' ')
    [ -z "$left" ] && { echo 0; return; }
    if [[ "$left" == *-* ]]; then d=${left%%-*}; left=${left#*-}; fi
    IFS=: read -r -a parts <<< "$left"
    case ${#parts[@]} in
        3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
        2) m=${parts[0]}; s=${parts[1]} ;;
        1) s=${parts[0]} ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

log "start on $(hostname), gpu=${CUDA_VISIBLE_DEVICES:-none}, checkpoint step=$(ckpt_step)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

if [ -f "$XQC_RUN_DIR/DONE" ]; then
    log "DONE already present, nothing to do"
    cancel_rest "run finished"
    exit 0
fi

STEP_BEFORE=$(ckpt_step)
LEFT=$(remaining_seconds)
# Fallback budget: stop 10 min before the wall limit even without the signal.
MAX_RUNTIME=$(( LEFT > 1200 ? LEFT - 600 : LEFT / 2 ))

mapfile -t OVERRIDES < <(grep -v '^\s*#' "$XQC_RUN_DIR/run.args" | sed '/^\s*$/d')
cd "$XQC_CODE"
python -u train_xqc.py "${OVERRIDES[@]}" run_dir="$XQC_RUN_DIR" max_runtime_s="$MAX_RUNTIME" &
PID=$!

forward() { log "received $1, forwarding to python pid $PID"; kill -"$1" "$PID" 2>/dev/null; }
trap 'forward USR1' USR1
trap 'forward TERM' TERM

# `wait` returns early (status > 128) when a trapped signal arrives; keep
# waiting until the Python process has really exited.
while true; do
    wait "$PID"
    RC=$?
    kill -0 "$PID" 2>/dev/null || break
done

STEP_AFTER=$(ckpt_step)
log "python exited rc=$RC, checkpoint step $STEP_BEFORE -> $STEP_AFTER"

if [ -f "$XQC_RUN_DIR/DONE" ]; then
    log "run finished"
    cancel_rest "run finished"
    exit 0
fi

FAILS_FILE="$CHAIN_DIR/consecutive_failures"
if [ "$RC" -eq 0 ] && [ "$STEP_AFTER" -gt "$STEP_BEFORE" ]; then
    echo 0 > "$FAILS_FILE"
    exit 0
fi

FAILS=$(( $(cat "$FAILS_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$FAILS" > "$FAILS_FILE"
if [ "$STEP_AFTER" -gt "$STEP_BEFORE" ]; then
    # Crashed, but progress was checkpointed: the next job resumes from there.
    echo 0 > "$FAILS_FILE"
    log "job failed (rc=$RC) after progress; next job will resume from step $STEP_AFTER"
    exit "$RC"
fi
if [ "$FAILS" -ge 2 ]; then
    log "ABORT: $FAILS consecutive jobs made no progress (last rc=$RC). Fix the cause, then relaunch the same run name to resume."
    cancel_rest "aborted after $FAILS failures"
    exit 1
fi
log "job made no progress (rc=$RC), failure $FAILS/2; next job in chain will retry"
exit "${RC:-1}"
