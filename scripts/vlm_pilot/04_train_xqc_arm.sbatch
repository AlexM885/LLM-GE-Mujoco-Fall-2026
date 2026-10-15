#!/bin/bash
# Train one XQC gene (seed network, no mutation) with `num_seeds` parallel
# training seeds, under one arm (xqc_vlm | xqc_control).
#
# XQC trains num_seeds seeds inside ONE sbatch job (batched replay buffer,
# shared GPU), so N=10 paired seeds = 1 sbatch per arm, not N sbatches.
# The paired-seed CRN is preserved by using the same -seed and -num_seeds
# on both arms; the only difference is whether LLMGE_AUX_REWARD_CACHE is
# set, which the make_env patch on this branch reads to decide whether to
# wrap the env with AuxRewardWrapper.
#
# Env in:
#   ARM                  "xqc_vlm" or "xqc_control"
#   VLM_PILOT_ROOT       shared scratch root for this pilot
#   PILOT_ENV            Gymnasium env id (default HalfCheetah-v4)
#   PILOT_TIMESTEPS      per-seed training budget (default 500000)
#   PILOT_NUM_SEEDS      number of parallel training seeds (default 10)
#   PILOT_BASE_SEED      base seed; XQC uses this + 42 for eval (default 0)
#   LLMGE_AUX_REWARD_*   aux reward cache + hyperparameter overrides,
#                        exported through to train_xqc.py's subprocess
#SBATCH --job-name=xqc_pilot_arm
#SBATCH -p ice-gpu
#SBATCH -t 2:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem 48G
#SBATCH -c 12
#SBATCH --output=run_job_outputs/vlm_pilot_xqc/arm_%j.out
set -euo pipefail

: "${ARM:?set ARM=xqc_vlm or ARM=xqc_control}"
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT}"
PILOT_ENV="${PILOT_ENV:-HalfCheetah-v4}"
PILOT_TIMESTEPS="${PILOT_TIMESTEPS:-500000}"
PILOT_NUM_SEEDS="${PILOT_NUM_SEEDS:-10}"
PILOT_BASE_SEED="${PILOT_BASE_SEED:-0}"

ROOT="$(pwd)"
mkdir -p "$VLM_PILOT_ROOT/$ARM" \
         "$VLM_PILOT_ROOT/${ARM}_runs" \
         run_job_outputs/vlm_pilot_xqc

hostname
module load cuda
nvidia-smi -L || echo "nvidia-smi not available (continuing)"

# Shazeb's XQC venv in scratch. env.sh activates the venv and points caches
# at scratch so nothing lands on the home quota.
source "$ROOT/sota/XQC/pace/env.sh"

# aux reward: only the vlm arm gets the cache set. Keep whatever the caller
# exported for the other LLMGE_AUX_REWARD_* overrides (lam, anneal, pbrs...).
if [ "$ARM" = "xqc_control" ]; then
    unset LLMGE_AUX_REWARD_CACHE || true
fi
echo "ARM=$ARM  env=$PILOT_ENV  cache=${LLMGE_AUX_REWARD_CACHE:-<none>}  " \
     "lam=${LLMGE_AUX_REWARD_LAM:-<cache>}  pbrs=${LLMGE_AUX_REWARD_PBRS:-0}"

# Seed-gene lives in Shazeb's XQC repo as llmge/network.py; the gene id in
# the standard LLM-GE contract is `models.network_<id>`.
GENE_ID="xqc_pilot_seed"
cp -f sota/XQC/llmge/network.py "sota/XQC/llmge/models/network_${GENE_ID}.py" 2>/dev/null || \
  (mkdir -p sota/XQC/llmge/models && cp -f sota/XQC/llmge/network.py \
    "sota/XQC/llmge/models/network_${GENE_ID}.py")
touch sota/XQC/llmge/models/__init__.py

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

OUT_JSON="$VLM_PILOT_ROOT/$ARM/results.json"
RUN_DIR="$VLM_PILOT_ROOT/${ARM}_runs/$GENE_ID"

cd "$ROOT/sota/MujocoRL"
# train_gene_xqc.py (MESB wrapper) invokes train_xqc.py as a subprocess;
# both inherit the LLMGE_AUX_REWARD_* env vars, which xqc/envs/envs.py's
# _maybe_wrap_aux_reward reads inside make_env.
python train_gene_xqc.py \
  --gene-file "$ROOT/sota/XQC/llmge/models/network_${GENE_ID}.py" \
  --gene-id "$GENE_ID" \
  --out-json "$OUT_JSON" \
  --run-dir "$RUN_DIR" \
  --generation 0 \
  --env "$PILOT_ENV" \
  --seed "$PILOT_BASE_SEED" \
  --timesteps "$PILOT_TIMESTEPS" \
  --num-seeds "$PILOT_NUM_SEEDS" \
  --eval-episodes 10

echo "Arm=$ARM done -> $OUT_JSON"
