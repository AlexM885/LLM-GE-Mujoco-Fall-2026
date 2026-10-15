#!/usr/bin/env bash
# Local smoke of the VLM aux-reward pipeline: no PACE, no GPU, no sb3.
#
# Hand-writes a sandbox-valid aux_reward cache, confirms the trainer-side
# loader accepts it, writes synthetic per-seed stats for 5 seeds x 2 arms,
# runs the paired-seed report, and prints it.
#
# Takes a few seconds. If this fails, don't spend PACE time; the schema
# contract between the modules has drifted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SMOKE_ROOT="$(mktemp -d -t vlm-pilot-smoke-XXXXXX)"
echo "Smoke root: $SMOKE_ROOT"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/nix/store/3b7njvjfc2w5lhj1jklrsw4lq7nwypri-gcc-15.2.0-lib/lib:/nix/store/wgw25d181h0qks7qsfp1bh7c5nd3f0s7-zlib-1.3.2/lib"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "--- step 1: write a hand-crafted sandbox-valid aux_reward cache ---"
cat > "$SMOKE_ROOT/aux_cache.json" <<'JSON'
{
  "schema": "aux-reward-cache-v1",
  "aux_reward_code": "def aux_reward(obs, action, info):\n    symmetry = -float(np.sum(np.abs(obs[2:5] - obs[5:8])))\n    wobble = -float(np.square(obs[10]))\n    return symmetry + 0.5 * wobble\n",
  "lam": 0.1,
  "anneal_steps": 50000,
  "rationale": "smoke: punish asymmetric gait and torso wobble",
  "generation": 0,
  "validation": "ok"
}
JSON

echo "--- step 2: loader accepts the cache ---"
"$PY" - <<PY
import sys, os
sys.path.insert(0, "$ROOT/sota/MujocoRL")
import train_rl_vlm as trv
import numpy as np
r = trv._load_aux_reward_cache("$SMOKE_ROOT/aux_cache.json")
assert r is not None, "loader rejected a valid cache"
fn, lam, anneal, meta = r
assert callable(fn) and 0 <= lam <= 1
# Call it to confirm the sandbox returns a float
v = fn(np.zeros(17), np.zeros(6), {"reward_forward": 0.0,
                                    "reward_survive": 1.0,
                                    "reward_ctrl": 0.0,
                                    "x_position": 0.0})
print("  loaded ok, lam=%s anneal=%s aux(zero_obs)=%.4f" % (lam, anneal, v))
PY

echo "--- step 3: write synthetic per-seed stats for 5 seeds x 2 arms ---"
mkdir -p "$SMOKE_ROOT/vlm" "$SMOKE_ROOT/control"
"$PY" - <<PY
import json, os, random
random.seed(0)
root = "$SMOKE_ROOT"
# VLM arm has an edge; control is noisier around 2400
for seed in range(5):
    base = 2400 + random.gauss(0, 30)
    vlm = {"seed": seed, "best_reward": base + 60 + random.gauss(0, 30),
           "mean_reward": base + 40 + random.gauss(0, 20)}
    ctrl = {"seed": seed, "best_reward": base + random.gauss(0, 30),
            "mean_reward": base + random.gauss(0, 20)}
    with open(os.path.join(root, "vlm", f"seed_{seed}.json"), "w") as f:
        json.dump(vlm, f)
    with open(os.path.join(root, "control", f"seed_{seed}.json"), "w") as f:
        json.dump(ctrl, f)
PY

echo "--- step 4: render the paired-seed report ---"
"$PY" "$ROOT/analysis/scripts/vlm_aux_reward_report.py" \
    --vlm-arm "$SMOKE_ROOT/vlm" \
    --control-arm "$SMOKE_ROOT/control" \
    --metric best_reward \
    --out - \
    --notes "SMOKE RUN - synthetic data, do not use as evidence"

echo
echo "Smoke pilot OK."
echo "Scratch kept at $SMOKE_ROOT (delete when done)"
