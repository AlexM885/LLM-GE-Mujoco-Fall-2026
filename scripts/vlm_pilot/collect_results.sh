#!/usr/bin/env bash
# Gather per-seed stats from each arm's stats_<arm>/seed_<n>/ directory and
# emit the N-arm paired-seed report. Idempotent; safe to run while arms are
# still in flight — only completed seeds are included.
#
# Env in:
#   VLM_PILOT_ROOT   shared scratch root
#   ARMS             comma-separated list of arm names (default: control,vlm_mock,vlm)
#   METRIC           field to compare (default: best_reward)
set -euo pipefail
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT first}"
ARMS="${ARMS:-control,vlm_mock,vlm}"
METRIC="${METRIC:-best_reward}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# For each arm, collate the per-seed stats JSONs from stats_<arm>/seed_<n>/
# into a single per-arm directory of normalised per-seed records.
IFS=',' read -r -a ARM_ARR <<< "$ARMS"
REPORT_ARGS=()
GENE_ID_DEFAULT="vlm_pilot_seed"
for arm in "${ARM_ARR[@]}"; do
    out="$VLM_PILOT_ROOT/$arm"
    mkdir -p "$out"
    stats_root="$VLM_PILOT_ROOT/stats_${arm}"
    [ -d "$stats_root" ] || { echo "WARN: $stats_root missing, skipping $arm"; continue; }
    for seed_dir in "$stats_root"/seed_*; do
        [ -d "$seed_dir" ] || continue
        seed=$(basename "$seed_dir" | sed 's/^seed_//')
        # Find the newest *_stats.json in this seed dir
        stats=$(ls -t "$seed_dir"/*_stats.json 2>/dev/null | head -1)
        [ -n "$stats" ] || continue
        row="$out/seed_$seed.json"
        python3 - "$stats" "$row" "$seed" <<'PY'
import json, sys
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f: s = json.load(f)
row = {
    "seed": seed,
    "best_reward": s.get("mean_reward"),
    "per_episode_rewards": s.get("per_episode_rewards", []),
    "foot_contact_0": s.get("foot_contact_0"),
    "foot_contact_1": s.get("foot_contact_1"),
    "mean_distance": s.get("mean_distance"),
    "aux_reward": s.get("aux_reward"),
    "aux_reward_audit": s.get("aux_reward_audit"),
    "train_time_sec": s.get("train_time_sec"),
    "score_band": s.get("score_band"),
    "model_path": s.get("model_path"),
}
with open(dst, "w") as f: json.dump(row, f, indent=2)
PY
    done
    REPORT_ARGS+=(--arm "$arm=$out")
done

# Default: compare each non-first arm to the first.
echo "Collected: ${REPORT_ARGS[*]}"
echo
mkdir -p "$VLM_PILOT_ROOT/report"
REPORT_PATH="$VLM_PILOT_ROOT/report/pilot_report.md"
env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env \
    python analysis/scripts/vlm_aux_reward_report.py \
    "${REPORT_ARGS[@]}" \
    --metric "$METRIC" \
    --out "$REPORT_PATH" \
    --notes "Pilot 1g x N seeds x ${#ARM_ARR[@]} arms, Walker2d-v5 TD3, 250k steps per child, paired seeds (CRN)."

echo
echo "Report: $REPORT_PATH"
echo
cat "$REPORT_PATH"
