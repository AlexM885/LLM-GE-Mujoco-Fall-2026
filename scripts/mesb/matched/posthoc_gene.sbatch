#!/bin/bash
#SBATCH --job-name=posthoc-gene
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -t 3:00:00
#SBATCH -G 1
#SBATCH --mem 16G
#SBATCH -c 12
#SBATCH -N 1
# ---------------------------------------------------------------------------
# Post-hoc check of one XQC gene (retrain with new seeds, then reward under
# changed conditions and gait features). Same GPU settings as the gene jobs of
# the matched runs. Submitted by scripts/mesb/matched/posthoc_submit.py, which
# passes --output and the arguments of scripts/mesb/matched/posthoc_eval.py.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
echo "post-hoc gene check on $(hostname) | job ${SLURM_JOB_ID:-local}"
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU visible; JAX will run on CPU"
export XQC_REPO="$PWD"
set +u
# shellcheck disable=SC1091
source sota/XQC/pace/env.sh
set -u
python scripts/mesb/matched/posthoc_eval.py "$@"
