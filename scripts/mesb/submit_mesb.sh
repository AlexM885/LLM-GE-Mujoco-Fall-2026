#!/bin/bash
# Submit a MESB run to Slurm from the repository root.
#
#   scripts/mesb/submit_mesb.sh RUN_NAME [SEED] [SELECTION_MODE]
#
# Any other setting from run_mesb.sbatch can be passed as an environment
# variable, e.g.
#   DIMS_X=10 DIMS_Y=10 REMAP_FREQUENCY=50 scripts/mesb/submit_mesb.sh mesb_10x10 1
# Extra sbatch flags (partition, account, QoS) go in SBATCH_FLAGS:
#   SBATCH_FLAGS="-p ice-cpu -A my-account" scripts/mesb/submit_mesb.sh mesb_s0
#
# CHAIN=N submits N jobs back to back (each starts after the previous one
# ends, for any reason). Every job after the first resumes the same run from
# its last checkpoint, so a run longer than the 16-hour job limit continues
# automatically. A job that finds the run already complete only re-runs the
# analysis and exits.
#   CHAIN=3 scripts/mesb/submit_mesb.sh mesb_full_s0 0 mesb
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

RUN_NAME="${1:?usage: submit_mesb.sh RUN_NAME [SEED] [SELECTION_MODE]}"
export RUN_NAME
export SEED="${2:-${SEED:-0}}"
export SELECTION_MODE="${3:-${SELECTION_MODE:-mesb}}"
CHAIN="${CHAIN:-1}"

mkdir -p mujoco_rl_output/slurm_logs   # --output dir must exist before sbatch
prev=""
for i in $(seq 1 "$CHAIN"); do
    dep=()
    [[ -n "$prev" ]] && dep=(--dependency="afterany:${prev}")
    # shellcheck disable=SC2086
    prev=$(sbatch --parsable ${SBATCH_FLAGS:-} "${dep[@]}" --job-name "mesb-${RUN_NAME}" \
        --export=ALL scripts/mesb/run_mesb.sbatch)
    prev="${prev%%;*}"
    echo "submitted mesb-${RUN_NAME} part ${i}/${CHAIN}: job ${prev}"
done
