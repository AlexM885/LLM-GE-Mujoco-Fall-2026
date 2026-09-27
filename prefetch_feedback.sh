#!/bin/bash
# Submit the observer batch job for every parent currently in the archive.
#
# Usage:
#   ./prefetch_feedback.sh [checkpoint_dir] [visual|telemetry]
#
# Run this after a generation has trained checkpoints and before the next
# generation, so the cached observations already exist when mutations happen.
set -euo pipefail

CHECKPOINTS="${1:-mujoco_islands_run/island_llama3_Mujoco-Normal}"
MODE="${2:-${LLMGE_FEEDBACK_MODE:-visual}}"

if [ ! -d "$CHECKPOINTS" ]; then
    echo "checkpoint dir not found: $CHECKPOINTS" >&2
    exit 1
fi

GENES="$(uv run python src/feedback_prefetch.py --checkpoints "$CHECKPOINTS")"
if [ -z "$(echo "$GENES" | tr -d '[:space:]')" ]; then
    echo "No parent genes found in $CHECKPOINTS" >&2
    exit 1
fi

echo "Submitting observer job for parents:"
echo "$GENES" | tr ',' '\n'
sbatch observer.sh "$GENES" "$MODE"
