#!/bin/bash
# Train one gene with one seed, under one arm (vlm | control).
#
# Submit as a seed-indexed array:
#   ARM=vlm     sbatch --array=0-4 scripts/vlm_pilot/02_train_arm.sbatch
#   ARM=control sbatch --array=0-4 scripts/vlm_pilot/02_train_arm.sbatch
#
# Env in:
#   ARM              "vlm" or "control"
#   VLM_PILOT_ROOT   shared scratch root
#   PILOT_TIMESTEPS  training budget per child (default 250000)
#SBATCH --job-name=vlm_pilot_arm
#SBATCH -p ice-gpu
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH --output=run_job_outputs/vlm_pilot/arm_%A_%a.out
set -euo pipefail

: "${ARM:?set ARM=vlm or ARM=control}"
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT}"
PILOT_TIMESTEPS="${PILOT_TIMESTEPS:-250000}"
GENE_ID="vlm_pilot_seed"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

ROOT="$(pwd)"
mkdir -p "$VLM_PILOT_ROOT/$ARM" \
         "$VLM_PILOT_ROOT/trained_models_$ARM/seed_$SEED" \
         "$VLM_PILOT_ROOT/stats_$ARM/seed_$SEED"

mkdir -p run_job_outputs/vlm_pilot
hostname
module load cuda
module load uv
nvidia-smi -L || echo "nvidia-smi not available (continuing)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

# Both arms run the SAME binary. The only difference is the aux cache env.
# control:  no cache -> vanilla TD3
# vlm:      cache from VLM observer at $VLM_PILOT_ROOT/aux_reward_cache.json
# vlm_mock: cache passed in via LLMGE_AUX_REWARD_CACHE (e.g. a hand-crafted one)
# any other arm: whatever the caller exports.
if [ "$ARM" = "control" ]; then
    unset LLMGE_AUX_REWARD_CACHE || true
elif [ "$ARM" = "vlm" ]; then
    export LLMGE_AUX_REWARD_CACHE="$VLM_PILOT_ROOT/aux_reward_cache.json"
fi
echo "ARM=$ARM, cache=${LLMGE_AUX_REWARD_CACHE:-<none>}"

# Pin all randomness: numpy, torch, env.first_reset, action noise.
export PYTHONHASHSEED="$SEED"
export MUJOCO_SEED="$SEED"

cd "$ROOT/sota/MujocoRL"
# Each seed needs a unique gene module name so sb3's save/load paths don't
# collide across the array; copy the shared seed network per seed.
GENE_SEED_ID="${GENE_ID}_s${SEED}"
mkdir -p models
cp -f "network.py" "models/network_${GENE_SEED_ID}.py"
# sb3 needs models/__init__.py so importlib.import_module("models.*") works
touch models/__init__.py

env -u VIRTUAL_ENV uv run --isolated --project eval_env python train_rl_vlm.py \
    -network "models.network_${GENE_SEED_ID}" \
    -env "${PILOT_ENV:-HalfCheetah-v4}" \
    -timesteps "$PILOT_TIMESTEPS" \
    -model_dir "$VLM_PILOT_ROOT/trained_models_$ARM/seed_$SEED" \
    -stats_dir "$VLM_PILOT_ROOT/stats_$ARM/seed_$SEED"

# Normalise the per-seed stats JSON into the shape the report expects.
cd "$ROOT"
python3 - <<PY
import json, os
src = "$VLM_PILOT_ROOT/stats_$ARM/seed_$SEED/${GENE_SEED_ID}_stats.json"
with open(src) as f: s = json.load(f)
row = {
    "seed": $SEED,
    "best_reward": s.get("mean_reward"),
    "per_episode_rewards": s.get("per_episode_rewards", []),
    "foot_contact_0": s.get("foot_contact_0"),
    "foot_contact_1": s.get("foot_contact_1"),
    "mean_distance": s.get("mean_distance"),
    "aux_reward": s.get("aux_reward"),
    "aux_reward_audit": s.get("aux_reward_audit"),
    "train_time_sec": s.get("train_time_sec"),
    "score_band": s.get("score_band"),
}
out = "$VLM_PILOT_ROOT/$ARM/seed_$SEED.json"
with open(out, "w") as f: json.dump(row, f, indent=2)
print(f"wrote {out}")
PY
echo "Arm=$ARM seed=$SEED done"
