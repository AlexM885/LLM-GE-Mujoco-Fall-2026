#!/bin/bash
#SBATCH --job-name=mesb-xqc-gene
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -t 1:30:00
#SBATCH -G 1
#SBATCH --mem 16G
#SBATCH -c 12
#SBATCH -N 1
# ---------------------------------------------------------------------------
# One XQC gene for MESB: train + evaluate on one GPU, write the MESB result
# JSON. Submitted by run_mesb.py --algo xqc --evaluator slurm (one job per
# gene, sbatch --wait); the driver passes --output and the arguments of
# sota/MujocoRL/train_gene_xqc.py.
#
# Resources are those of Shazeb's XQC evaluation jobs (slurm-config/
# slurm_config.yaml on shazeb/halfcheetah-xqc): an NVIDIA GPU on ice-gpu
# (AMD MI210 nodes make the CUDA JAX build fall back to CPU), 1.5 h.
# The XQC environment comes from sota/XQC/pace/env.sh (XQC_BASE, default
# ~/scratch/xqc-baseline; install once with sota/XQC/pace/install_env.sh).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
echo "MESB XQC gene on $(hostname) | job ${SLURM_JOB_ID:-local}"

if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU visible; JAX will run on CPU"

export XQC_REPO="$PWD"
set +u   # env.sh and the venv activate script may read unset variables
# shellcheck disable=SC1091
source sota/XQC/pace/env.sh
set -u
python sota/MujocoRL/train_gene_xqc.py "$@"
