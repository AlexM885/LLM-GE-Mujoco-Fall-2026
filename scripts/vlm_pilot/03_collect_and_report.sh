#!/usr/bin/env bash
# Gather per-seed stats for both arms and render the paired-seed report.
set -euo pipefail
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT first}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "VLM arm seeds:"
ls "$VLM_PILOT_ROOT/vlm/" 2>/dev/null | sort
echo
echo "Control arm seeds:"
ls "$VLM_PILOT_ROOT/control/" 2>/dev/null | sort

uv run python analysis/scripts/vlm_aux_reward_report.py \
    --vlm-arm "$VLM_PILOT_ROOT/vlm" \
    --control-arm "$VLM_PILOT_ROOT/control" \
    --metric best_reward \
    --out "$VLM_PILOT_ROOT/report.md" \
    --notes "Pilot: 1 gene x 5 seeds x 2 arms, 250k steps per child, Walker2d-v5 TD3."

echo
cat "$VLM_PILOT_ROOT/report.md"
