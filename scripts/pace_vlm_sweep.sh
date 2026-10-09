#!/bin/bash
# Sequentially run several MuJoCo VLM iteration arms on PACE.
#
# This avoids launching multiple 70B LLM servers at once. After each arm, it
# cancels that arm's server and writes a cumulative report.
set -euo pipefail

ROOT="${1:-/storage/ice1/5/8/nsani3/llmge-vlm-iterate-$(date +%Y%m%d)}"
SOURCE_RUN="${2:-/storage/ice1/5/8/nsani3/llmge-vlm-highbudget-20260927-173532}"
ARMS=(${LLMGE_SWEEP_ARMS:-planner_top32 actionable_top32 planner_top64_uniform telemetry_top})

if [[ "$ROOT" != /storage/ice1/* ]]; then
    echo "Refusing to run outside PACE scratch: $ROOT" >&2
    exit 2
fi

if [[ ! -f "$ROOT/.hf_token" ]]; then
    echo "Missing $ROOT/.hf_token. Create it with chmod 600 before starting." >&2
    exit 2
fi

job_field() {
    local arm_dir="$1"
    local key="$2"
    python3 - "$arm_dir/arm_jobs.json" "$key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
print(json.load(open(path)).get(key, ""))
PY
}

wait_for_job() {
    local job="$1"
    local label="$2"
    while squeue -j "$job" -h >/tmp/llmge_squeue_$$ 2>/dev/null && [[ -s /tmp/llmge_squeue_$$ ]]; do
        echo "[$(date -Is)] waiting for $label job $job"
        sleep 300
    done
    rm -f /tmp/llmge_squeue_$$
}

write_report() {
    python3 analysis/scripts/mujoco_vlm_iteration_report.py "$ROOT" \
        --include-known \
        --json-out "$ROOT/mujoco_vlm_iteration_report.json" \
        --md-out "$ROOT/mujoco_vlm_iteration_report.md" >/dev/null
    echo "Report: $ROOT/mujoco_vlm_iteration_report.md"
}

for arm in "${ARMS[@]}"; do
    echo "============================================================"
    echo "Launching arm: $arm"
    echo "============================================================"
    bash scripts/pace_vlm_iterate.sh "$arm" "$SOURCE_RUN" "$ROOT"
    ARM_DIR="$ROOT/$arm"
    RUN_JOB="$(job_field "$ARM_DIR" run)"
    SERVER_JOB="$(job_field "$ARM_DIR" server)"
    OBS_JOB="$(job_field "$ARM_DIR" observer)"

    echo "observer=$OBS_JOB run=$RUN_JOB server=$SERVER_JOB"
    wait_for_job "$RUN_JOB" "$arm run"
    if [[ -n "$SERVER_JOB" ]]; then
        scancel "$SERVER_JOB" 2>/dev/null || true
    fi
    write_report
done

rm -f "$ROOT/.hf_token"
write_report
echo "Sweep complete."
