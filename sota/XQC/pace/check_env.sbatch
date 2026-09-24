#!/bin/bash
# Environment check on a compute node. Output goes to $XQC_BASE/checks/.
#   GPU: sbatch -p ice-gpu --gres=gpu:1 sota/XQC/pace/check_env.sbatch
#   CPU: sbatch -p ice-cpu sota/XQC/pace/check_env.sbatch --no-gpu-required
#SBATCH --job-name=xqc_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
set -euo pipefail
source "${XQC_REPO:-$(readlink -f "$HOME/scratch")/xqc-baseline/repo}/sota/XQC/pace/env.sh"
mkdir -p "$XQC_BASE/checks"
exec > "$XQC_BASE/checks/check_env_${SLURM_JOB_ID}.out" 2>&1
echo "node=$(hostname) partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK JAX_PLATFORMS=${JAX_PLATFORMS:-auto}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia gpu"
cd "$XQC_CODE"
python pace/check_env.py "$@"
