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
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

RUN_NAME="${1:?usage: submit_mesb.sh RUN_NAME [SEED] [SELECTION_MODE]}"
export RUN_NAME
export SEED="${2:-${SEED:-0}}"
export SELECTION_MODE="${3:-${SELECTION_MODE:-mesb}}"

mkdir -p mujoco_rl_output/slurm_logs   # --output dir must exist before sbatch
# shellcheck disable=SC2086
sbatch ${SBATCH_FLAGS:-} --job-name "mesb-${RUN_NAME}" --export=ALL scripts/mesb/run_mesb.sbatch
