#!/bin/bash
# Keep one LLM-GE island running for days: a chain of LLM-server jobs (server.sh,
# 8 h, 2 GPUs) and a chain of controller jobs (run.sh on ice-cpu), each job queued
# afterany its predecessor. Every controller job resumes the run from its newest
# checkpoint, so set LLMGE_MIGRATION_GEN=0 (no stop at migration boundaries).
#
#   scripts/llmge_chain.sh RUN SERVER_JOBS CONTROLLER_JOBS [CONTROLLER_TIME]
#
# CONTROLLER_TIME defaults to 18:00:00 (the ice-cpu limit): a controller killed at
# its limit loses the generation it was in, so long controller jobs waste less.
# The current environment goes to every job (sbatch --export=ALL), e.g.
#   LLMGE_ALGO=xqc LLMGE_ARCHIVE=mome LLMGE_POP_SIZE=16 LLMGE_NUM_GENERATIONS=200 \
#   LLMGE_MIGRATION_GEN=0 LLMGE_GENE_TIME=2:00:00 scripts/llmge_chain.sh xqc_mome_run1 21 10
# Writes the job ids to RUN/chain_jobs.txt. Stop the run with scancel on those ids.
#
# The first controller starts only once the first server has started (--dependency
# after:), and controllers wait up to LLM_SERVER_READY_TIMEOUT (default 8 h) for a
# server, so a queued or restarting server never makes gene creation fail.
# SERVER_JOBS=0 with LLMGE_AFTER_SERVER=<job id> chains only controllers, starting
# after that (already queued) server job.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

[ $# -ge 3 ] || { sed -n '2,20p' "$0"; exit 1; }
RUN="$1"; SERVERS="$2"; CONTROLLERS="$3"; CTIME="${4:-18:00:00}"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run name: $RUN" >&2; exit 1; }
[ "${LLMGE_MIGRATION_GEN:-5}" = 0 ] || echo "WARNING: LLMGE_MIGRATION_GEN=${LLMGE_MIGRATION_GEN:-5} (unset = 5): each controller stops at the next multiple of it" >&2
export LLMGE_RUN="$RUN"
export LLM_SERVER_READY_TIMEOUT="${LLM_SERVER_READY_TIMEOUT:-28800}"
mkdir -p "$RUN" run_job_outputs/islands run_job_outputs/server

chain() {  # chain <count> <first job's dependency or ""> <sbatch args...>; prints the job ids
    local prev="" id first="$2"
    for _ in $(seq 1 "$1"); do
        local dep=()
        if [ -n "$prev" ]; then dep=(--dependency="afterany:$prev")
        elif [ -n "$first" ]; then dep=(--dependency="$first"); fi
        id=$(sbatch --parsable "${dep[@]}" "${@:3}")
        prev="${id%%;*}"
        echo "$prev"
    done
}

SERVER_IDS=$(chain "$SERVERS" "" server.sh)
FIRST_SERVER="${LLMGE_AFTER_SERVER:-$(echo $SERVER_IDS | cut -d' ' -f1)}"
CONTROLLER_IDS=$(chain "$CONTROLLERS" "${FIRST_SERVER:+after:$FIRST_SERVER}" -p ice-cpu -t "$CTIME" run.sh)
{
    echo "# $(date -Is) commit $(git rev-parse --short HEAD)"
    env | grep -E '^(LLMGE|XQC)_|^LLM_SERVER_READY_TIMEOUT' | sort | sed 's/^/# /'
    echo "servers: $(echo $SERVER_IDS)"
    echo "controllers: $(echo $CONTROLLER_IDS)"
} | tee -a "$RUN/chain_jobs.txt"
echo "stop everything: scancel $(echo $SERVER_IDS $CONTROLLER_IDS)"
