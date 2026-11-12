#!/bin/bash
# Submit with: ./run_AgenticIslands.sh
# Override paths/settings with environment variables, for example:
# ./run_AgenticIslands.sh /path/to/cifar10

#SBATCH --job-name=agentic-islands
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --output=slurm-agentic-islands-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch was not found. Run this command on a SLURM login node." >&2
    exit 127
  fi
  exec sbatch "$SCRIPT_DIR/run_AgenticIslands.sh"
fi

ROOT_DIR="$SCRIPT_DIR"
DATA_PATH="${DATA_PATH:-}"
GENERATIONS="${GENERATIONS:-2}"
EPOCHS="${EPOCHS:-2}"
POPULATION_SIZE="${POPULATION_SIZE:-4}"
CANDIDATES_PER_CYCLE="${CANDIDATES_PER_CYCLE:-1}"
BATCH_SIZE="${BATCH_SIZE:-216}"
WORKERS="${WORKERS:-4}"
INFERENCE_SUBMISSION="${INFERENCE_SUBMISSION:-true}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Error: dataset directory does not exist: ${DATA_PATH:-<not set>}" >&2
  exit 2
fi

module load cuda
module load anaconda3
export CUDA_VISIBLE_DEVICES=0

CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-}}"
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
  else
    echo "Error: Conda environment '$CONDA_ENV' was not found." >&2
    echo "Set CONDA_ENV to an environment listed by 'conda info --envs'." >&2
    exit 1
  fi
else
  echo "No CONDA_ENV was set; using the Python available after module loading."
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

INFERENCE_FLAG="--no-inference-submission"
if [[ "${INFERENCE_SUBMISSION,,}" == "true" ]]; then
  INFERENCE_FLAG="--inference-submission"
fi

python "$ROOT_DIR/run_AgenticIslands.py" \
  --root "$ROOT_DIR" \
  --data "$DATA_PATH" \
  --generations "$GENERATIONS" \
  --epochs "$EPOCHS" \
  --population-size "$POPULATION_SIZE" \
  --candidates-per-cycle "$CANDIDATES_PER_CYCLE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  "$INFERENCE_FLAG" \
  --output "$ROOT_DIR/agentic_islands_results_${SLURM_JOB_ID:-local}.json"
