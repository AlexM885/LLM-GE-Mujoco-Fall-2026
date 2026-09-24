#!/bin/bash
# Launch (or extend) a chain of dependent Slurm jobs for one XQC run.
#
#   pace/launch_chain.sh RUN_NAME [options] [-- hydra overrides...]
#
# Options:
#   -n, --jobs N         jobs in the chain (default 3)
#   -t, --time HH:MM:SS  wall time per job, max 08:00:00 (default 08:00:00)
#   -p, --partition P    Slurm partition (default $XQC_PARTITION)
#   -q, --qos Q          Slurm QoS (default $XQC_QOS)
#   -g, --gres G         GPU request (default $XQC_GRES or gpu:1)
#   --dry-run            print the sbatch commands without submitting
#
# First launch of RUN_NAME stores the Hydra overrides in $XQC_RUNS/RUN_NAME/run.args.
# Relaunching an existing RUN_NAME (no overrides) resumes it from its latest
# checkpoint; giving overrides that differ from run.args is an error.
#
# Example:
#   pace/launch_chain.sh halfcheetah_v4_s0 -n 3 -- +experiment=halfcheetah_v4_baseline
#
# Dependency type: afterany. afterok would stop the chain on any non-zero exit,
# including a node failure or crash that the next job could simply resume from.
# With afterany the chain continues, and the job script itself stops the chain
# when the run is DONE or after 2 consecutive jobs that made no progress.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XQC_REPO="${XQC_REPO:-$(cd "$HERE/../../.." && pwd)}"
# shellcheck disable=SC1091
source "$HERE/env.sh"

usage() { sed -n '2,24p' "$0"; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
[[ "$1" == -h || "$1" == --help ]] && usage 0
RUN_NAME="$1"; shift
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run name: $RUN_NAME" >&2; exit 1; }

JOBS=3; WALL="08:00:00"; DRY=0
PARTITION="${XQC_PARTITION:-}"; QOS="${XQC_QOS:-}"; GRES="${XQC_GRES:-gpu:1}"
OVERRIDES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--jobs) JOBS="$2"; shift 2 ;;
        -t|--time) WALL="$2"; shift 2 ;;
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -q|--qos) QOS="$2"; shift 2 ;;
        -g|--gres) GRES="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --) shift; OVERRIDES=("$@"); break ;;
        *) echo "unknown option $1" >&2; usage 1 ;;
    esac
done

# Enforce the 8 h cap.
IFS=: read -r H M S <<< "$WALL"
SECS=$(( 10#$H * 3600 + 10#$M * 60 + 10#${S:-0} ))
if [ "$SECS" -gt $(( 8 * 3600 )) ]; then echo "wall time $WALL exceeds 08:00:00" >&2; exit 1; fi
if [ "$SECS" -le 600 ]; then echo "note: wall time $WALL is shorter than the 300 s pre-timeout signal margin plus startup; fine for tests only" >&2; fi

RUN_DIR="$XQC_RUNS/$RUN_NAME"
mkdir -p "$RUN_DIR/chain" "$RUN_DIR/slurm"
ARGS_FILE="$RUN_DIR/run.args"
if [ -f "$ARGS_FILE" ]; then
    if [ ${#OVERRIDES[@]} -gt 0 ] && ! diff -q <(printf '%s\n' "${OVERRIDES[@]}") <(grep -v '^#' "$ARGS_FILE") >/dev/null; then
        echo "run $RUN_NAME already exists with different overrides:" >&2
        grep -v '^#' "$ARGS_FILE" >&2
        echo "relaunch without overrides to resume it, or pick a new run name" >&2
        exit 1
    fi
    echo "resuming existing run $RUN_NAME (latest checkpoint: $(cat "$RUN_DIR/checkpoints/latest" 2>/dev/null || echo none))"
else
    { echo "# Hydra overrides for run $RUN_NAME, created $(date '+%F %T')"; printf '%s\n' "${OVERRIDES[@]}"; } > "$ARGS_FILE"
    echo "new run $RUN_NAME"
fi
if [ -f "$RUN_DIR/DONE" ]; then echo "run $RUN_NAME is already DONE"; exit 0; fi

# If part of this chain is still queued or running, append after its last job.
PREV=""
if [ -f "$RUN_DIR/chain/jobs.txt" ]; then
    ACTIVE=$(squeue -h -u "$USER" -o %i 2>/dev/null | grep -Fxf "$RUN_DIR/chain/jobs.txt" | sort -n | tail -1 || true)
    [ -n "$ACTIVE" ] && PREV="$ACTIVE" && echo "chain has active job $PREV, appending after it"
fi
echo 0 > "$RUN_DIR/chain/consecutive_failures"

SBATCH_OPTS=(--job-name="xqc_${RUN_NAME}" --time="$WALL" --gres="$GRES"
    --output="$RUN_DIR/slurm/%j.out" --error="$RUN_DIR/slurm/%j.out"
    --export=ALL,XQC_RUN_DIR="$RUN_DIR",XQC_BASE="$XQC_BASE",XQC_REPO="$XQC_REPO")
[ -n "$PARTITION" ] && SBATCH_OPTS+=(--partition="$PARTITION")
[ -n "$QOS" ] && SBATCH_OPTS+=(--qos="$QOS")

for k in $(seq 1 "$JOBS"); do
    DEP=()
    [ -n "$PREV" ] && DEP=(--dependency="afterany:$PREV")
    CMD=(sbatch --parsable "${SBATCH_OPTS[@]}" "${DEP[@]}" "$XQC_CODE/pace/xqc_job.sbatch")
    if [ "$DRY" -eq 1 ]; then
        echo "${CMD[*]}"; PREV="DRYRUN$k"; continue
    fi
    JID=$("${CMD[@]}")
    JID="${JID%%;*}"
    echo "$JID" >> "$RUN_DIR/chain/jobs.txt"
    echo "$(date '+%F %T') submitted $JID dep=${PREV:-none}" >> "$RUN_DIR/chain/history.log"
    echo "submitted job $k/$JOBS: $JID (after ${PREV:-none})"
    PREV="$JID"
done

echo "run dir:  $RUN_DIR"
echo "monitor:  squeue -u $USER;  tail -f $RUN_DIR/slurm/<jobid>.out;  cat $RUN_DIR/summary.json"
echo "stop all: scancel \$(cat $RUN_DIR/chain/jobs.txt)"
