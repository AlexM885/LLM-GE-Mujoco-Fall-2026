#!/usr/bin/env bash
# Collect per-seed stats from an XQC pilot (one results.json per arm, each
# containing per_seed_mean_reward) and emit a paired-seed report in the same
# format as the TD3 pipeline so the same analysis script can consume it.
#
# Env in:
#   VLM_PILOT_ROOT   shared scratch root
#   ARMS             comma-separated list of arm names (default: xqc_control,xqc_vlm)
#   METRIC           field to compare (default: best_reward -- aliases to the
#                    per-seed mean reward written by train_gene_xqc.py)
set -euo pipefail
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT first}"
ARMS="${ARMS:-xqc_control,xqc_vlm}"
METRIC="${METRIC:-best_reward}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# For each arm, split results.json's per_seed_mean_reward into per-seed JSONs
# in the exact shape vlm_aux_reward_report.py expects (one file per seed with
# best_reward and seed fields).
IFS=',' read -r -a ARM_ARR <<< "$ARMS"
REPORT_ARGS=()
for arm in "${ARM_ARR[@]}"; do
    src="$VLM_PILOT_ROOT/$arm/results.json"
    out="$VLM_PILOT_ROOT/$arm"
    [ -f "$src" ] || { echo "WARN: $src missing, skipping $arm"; continue; }
    python3 - "$src" "$out" <<'PY'
import json, os, sys
src, out = sys.argv[1], sys.argv[2]
with open(src) as f: payload = json.load(f)
per_seed = payload.get("per_seed_mean_reward") or []
for i, r in enumerate(per_seed):
    row = {
        "seed": i,
        "best_reward": float(r),
        "mean_reward_arm": float(payload.get("mean_reward", 0)),
        "std_reward_arm": float(payload.get("std_reward", 0)),
        "train_time_sec": float(payload.get("train_time_sec", 0)),
        "mean_distance": float(payload.get("behavior", {}).get("mean_distance", 0)),
        "mean_control_cost": float(payload.get("behavior", {}).get("mean_control_cost", 0)),
        "algorithm": payload.get("algorithm", ""),
        "env_id": payload.get("env_id", ""),
    }
    with open(os.path.join(out, f"seed_{i}.json"), "w") as g: json.dump(row, g, indent=2)
print(f"[{out}] wrote {len(per_seed)} seed_*.json rows")
PY
    REPORT_ARGS+=(--arm "$arm=$out")
done

mkdir -p "$VLM_PILOT_ROOT/report"
REPORT_PATH="$VLM_PILOT_ROOT/report/xqc_pilot_report.md"
echo "Collected: ${REPORT_ARGS[*]}"
echo
env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env \
    python analysis/scripts/vlm_aux_reward_report.py \
    "${REPORT_ARGS[@]}" \
    --metric "$METRIC" \
    --out "$REPORT_PATH" \
    --notes "XQC HalfCheetah-v4 pilot, 1 gene x num_seeds parallel training seeds, paired-seed CRN."

echo
echo "Report: $REPORT_PATH"
echo
cat "$REPORT_PATH"
