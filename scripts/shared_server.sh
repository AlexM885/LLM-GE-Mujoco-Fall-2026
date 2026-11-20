#!/bin/bash
# One LLM server for the whole team: two chains of 8 h server jobs (server.sh), the second
# starting OFFSET hours after the first. Both write the same hostname file, so it always
# names the newest server, which has the most time left. A gap only happens if a new server
# job waits in the queue longer than OFFSET, instead of every 8 h.
#
#   scripts/shared_server.sh <jobs per chain> [offset hours, default 4]
#
# Runs point at it with:
#   LLMGE_HOSTNAME_FILE=$SHARED_DIR/hostname.log LLMGE_SHARED_PORT=$PORT scripts/tiered_run.sh <tier> <repeat>
#
# Environment: SHARED_DIR (default ~/scratch/llmge-shared; must be readable by teammates,
# see docs/TIERED_RUNS.md), LLMGE_SHARED_PORT (default 8742).
# Two chains use 2 x 2 GPUs x 8 h = 32 GPU-hours, the whole per-user cap: run genes elsewhere.
# Job ids go to $SHARED_DIR/server_jobs.txt. Stop: scancel $(cut -d' ' -f2 $SHARED_DIR/server_jobs.txt)
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

N="${1:-}"; OFFSET="${2:-4}"
[[ "$N" =~ ^[1-9][0-9]*$ ]] && [[ "$OFFSET" =~ ^[0-9]+$ ]] || { sed -n '2,16p' "$0"; exit 1; }
SHARED_DIR="${SHARED_DIR:-$(readlink -f "$HOME/scratch")/llmge-shared}"
export LLMGE_PORT="${LLMGE_SHARED_PORT:-8742}"
export LLMGE_HOSTNAME_FILE="$SHARED_DIR/hostname.log"
BAD_NODES="$(sed -n 's/^BAD_NODES="\([^"]*\)".*/\1/p' scripts/tiered_run.sh)"  # same list as the runs

umask 022                                   # hostname.log readable by teammates
mkdir -p "$SHARED_DIR" run_job_outputs/server
chmod 755 "$SHARED_DIR"

submit() {  # submit <chain> <extra sbatch args...>; prints the job id
    local id
    id=$(sbatch --parsable --exclude="$BAD_NODES" "${@:2}" server.sh)
    id="${id%%;*}"
    # PACE's submit plugin can replace --exclude on GPU jobs: set it again on the queued job.
    scontrol update JobId="$id" ExcNodeList="$BAD_NODES" 2>/dev/null || true
    echo "$1 $id" >> "$SHARED_DIR/server_jobs.txt"
    echo "$id"
}

echo "# $(date -Is) commit $(git rev-parse --short HEAD) port $LLMGE_PORT file $LLMGE_HOSTNAME_FILE" >> "$SHARED_DIR/server_jobs.txt"
for chain in A B; do
    prev=""
    for i in $(seq 1 "$N"); do
        if [ -n "$prev" ]; then prev=$(submit "$chain" --dependency="afterany:$prev")
        elif [ "$chain" = B ] && [ "$OFFSET" -gt 0 ]; then prev=$(submit "$chain" --begin="now+${OFFSET}hours")
        else prev=$(submit "$chain"); fi
    done
done
cat "$SHARED_DIR/server_jobs.txt"
echo
echo "Shared server: LLMGE_HOSTNAME_FILE=$LLMGE_HOSTNAME_FILE LLMGE_SHARED_PORT=$LLMGE_PORT"
echo "Check: squeue -u $USER -n LLMGE01_Server; curl -s -m 5 http://\$(cat $LLMGE_HOSTNAME_FILE):$LLMGE_PORT/"
