#!/bin/bash
# Launch one matched pair: shared generation 0, then the fixed-grid arm and
# the MESB arm side by side, each as CHAIN jobs that resume from checkpoints.
#
#   scripts/mesb/matched/launch_pair.sh PAIR [CHAIN]
#   scripts/mesb/matched/launch_pair.sh xqc_matched_p1 2
#
# Start the LLM server first (scripts/mesb/start_llm_server.sh N). Results go
# to mujoco_rl_output/matched/PAIR/{gen0,fixed,mesb}. Extra sbatch flags for
# these driver jobs go in SBATCH_FLAGS.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

PAIR="${1:?usage: launch_pair.sh PAIR [CHAIN]}"
CHAIN="${2:-2}"
mkdir -p mujoco_rl_output/slurm_logs

submit() {  # submit ROLE DEPENDENCY -> job id
    local role=$1 dep=$2 args=()
    [[ -n "$dep" ]] && args=(--dependency="$dep")
    # shellcheck disable=SC2086
    sbatch --parsable ${SBATCH_FLAGS:-} "${args[@]}" --job-name "matched-${PAIR}-${role}" \
        --export=ALL,PAIR="$PAIR",ROLE="$role" scripts/mesb/matched/matched_island.sbatch | cut -d';' -f1
}

gen0=$(submit gen0 "")
echo "generation 0:  job $gen0"
for role in fixed mesb; do
    prev=$(submit "$role" "afterok:$gen0")
    echo "$role arm part 1/$CHAIN: job $prev"
    for i in $(seq 2 "$CHAIN"); do
        prev=$(submit "$role" "afterany:$prev")
        echo "$role arm part $i/$CHAIN: job $prev"
    done
done
echo "watch: squeue -u \$USER ; tail -f mujoco_rl_output/slurm_logs/matched-${PAIR}-*.out"
