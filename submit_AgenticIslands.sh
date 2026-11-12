#!/bin/bash
# Submit the agentic-island SLURM job.
#
# Usage:
#   ./submit_AgenticIslands.sh /path/to/cifar10
#   GENERATIONS=2 EPOCHS=2 ./submit_AgenticIslands.sh /path/to/cifar10

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${1:-${DATA_PATH:-$ROOT_DIR/cifar10}}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch was not found. Run this on a SLURM login node." >&2
  exit 127
fi

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Error: dataset directory does not exist: $DATA_PATH" >&2
  echo "Usage: $0 /path/to/cifar10" >&2
  exit 2
fi

export DATA_PATH
export ROOT_DIR

echo "Submitting agentic-island job..."
echo "  Repository: $ROOT_DIR"
echo "  Dataset:    $DATA_PATH"
echo "  Generations: ${GENERATIONS:-2}"
echo "  Epochs:      ${EPOCHS:-2}"

exec sbatch "$ROOT_DIR/run_AgenticIslands.sh"
