#!/bin/bash
# Usage: ./run_AgenticIslands.sh /path/to/dataset
#SBATCH --job-name=agentic-islands
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|H200|V100-16GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-agentic-islands-%j.out

# ---------- Launcher ----------
if [[ -z "${AGENTIC_LAUNCHED:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DATA_PATH="${1:-${DATA_PATH:-}}"
  if [[ -z "$DATA_PATH" || ! -d "$DATA_PATH" ]]; then
    echo "Error: dataset directory missing. Usage: $0 /path/to/dataset" >&2
    exit 2
  fi
  command -v sbatch >/dev/null || { echo "sbatch not found; use a login node." >&2; exit 127; }
  exec sbatch --chdir="$SCRIPT_DIR" \
    --export=ALL,AGENTIC_LAUNCHED=1,ROOT_DIR="$SCRIPT_DIR",DATA_PATH="$(realpath "$DATA_PATH")" \
    "$SCRIPT_DIR/run_AgenticIslands.sh"
fi

# ---------- Job (compute node) ----------
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-$SLURM_SUBMIT_DIR}"     # NOT dirname of BASH_SOURCE
DATA_PATH="${DATA_PATH:?DATA_PATH not set}"
GENERATIONS="${GENERATIONS:-2}"
EPOCHS="${EPOCHS:-2}"
POPULATION_SIZE="${POPULATION_SIZE:-4}"
CANDIDATES_PER_CYCLE="${CANDIDATES_PER_CYCLE:-1}"
BATCH_SIZE="${BATCH_SIZE:-216}"
WORKERS="${WORKERS:-4}"
INFERENCE_SUBMISSION="${INFERENCE_SUBMISSION:-true}"
CONDA_ENV="${CONDA_ENV:-llm_guided_env}"

echo "Host: $(hostname)  Root: $ROOT_DIR  Data: $DATA_PATH"
cd "$ROOT_DIR"

set +u                                   # module/conda scripts use unset vars
module load cuda
source "$ROOT_DIR/.venv/bin/activate"
set -u

NVJIT="$(find "$ROOT_DIR/.venv" -type d -path '*nvidia/nvjitlink/lib' 2>/dev/null | head -n1)"
[[ -n "$NVJIT" ]] && export LD_LIBRARY_PATH="$NVJIT:${LD_LIBRARY_PATH:-}"

which python
python -c "import torch, deap; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$HOME/.conda/envs/$CONDA_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

INFERENCE_FLAG="--no-inference-submission"
[[ "${INFERENCE_SUBMISSION,,}" == "true" ]] && INFERENCE_FLAG="--inference-submission"

python run_AgenticIslands.py \
  --root "$ROOT_DIR" \
  --data "$DATA_PATH" \
  --generations "$GENERATIONS" \
  --epochs "$EPOCHS" \
  --population-size "$POPULATION_SIZE" \
  --candidates-per-cycle "$CANDIDATES_PER_CYCLE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  "$INFERENCE_FLAG" \
  --output "$ROOT_DIR/slurm-outs/agentic_islands_results_${SLURM_JOB_ID}.json"