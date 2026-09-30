#!/bin/bash
# Keep the local LLM server available for a long MESB run.
#
#   scripts/mesb/start_llm_server.sh [N]
#
# Submits N (default 3) server jobs back to back: each 8-hour job (the most
# the GPU QOS allows) is queued to start when the previous one ends. Every new
# server writes its node to hostname.log; MESB's LLM calls that hit the gap
# wait for it (up to LLM_SERVER_READY_TIMEOUT, 4 h by default) and retry, so
# the run does not need restarting. Cancel the rest with scancel when done.
#
# Extra sbatch flags go in SBATCH_FLAGS, e.g. SBATCH_FLAGS="-A <account>".
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

N="${1:-3}"
mkdir -p mujoco_rl_output/slurm_logs
prev=""
for i in $(seq 1 "$N"); do
    dep=()
    [[ -n "$prev" ]] && dep=(--dependency="afterany:${prev}")
    # shellcheck disable=SC2086
    prev=$(sbatch --parsable ${SBATCH_FLAGS:-} "${dep[@]}" scripts/mesb/llm_server.sbatch)
    prev="${prev%%;*}"
    echo "submitted LLM server part ${i}/${N}: job ${prev}"
done
echo "check with: squeue -u \$USER   |   curl http://\$(cat hostname.log):\$(uv run python -c 'import sys; sys.path.insert(0,\"src\"); from cfg import constants; print(constants.PORT)')/"
