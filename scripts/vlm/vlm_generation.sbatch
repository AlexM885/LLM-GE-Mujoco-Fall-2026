#!/bin/bash
#SBATCH --job-name=vlm_generation
#SBATCH -p ice-gpu
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C "gpu-h200|gpu-h100|gpu-a100|gpu-l40s|gpu-rtxpro-blackwell"
#SBATCH --mem 96G
#SBATCH -c 12
# ---------------------------------------------------------------------------
# One generation's VLM step (Neel's design), submitted by src/vlm_loop.py with
# sbatch --wait after the archive update:
#   1. Record rollouts (frames + telemetry) of the genes the VLM will watch,
#      the archive's best and worst, with Neel's capture scripts. A gene
#      recorded in an earlier generation is not recorded again.
#   2. Run Neel's aux_reward_observer.sh: the VLM watches the rollouts (and
#      its previous aux reward, for reflection) and writes this generation's
#      aux-reward file, which the next generation trains with.
#
# From the caller (src/vlm_loop.py):
#   VLM_PARENTS_FILE   lines "<gene_id> <td3|xqc> <model path>"
#   VLM_CAPTURE_ROOT   where captures go (one folder per gene)
#   VLM_ENV_ID         gymnasium env of the run
#   AUX_*              aux_reward_observer.sh's inputs (see that script)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
echo "VLM generation step on $(hostname) | job ${SLURM_JOB_ID:-local} | generation ${AUX_GENERATION:-?}"

: "${VLM_PARENTS_FILE:?}" "${VLM_CAPTURE_ROOT:?}" "${VLM_ENV_ID:?}"
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
    module load uv 2>/dev/null || true
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
mkdir -p "$VLM_CAPTURE_ROOT"

# 1. Captures. A failed capture only drops that gene from what the VLM sees.
while read -r GENE KIND MODEL; do
    [ -z "${GENE:-}" ] && continue
    OUT="$VLM_CAPTURE_ROOT/$GENE"
    if [ -f "$OUT/manifest.json" ] && [ -f "$OUT/telemetry.json" ]; then
        echo "[capture] $GENE already recorded"
        continue
    fi
    echo "[capture] $GENE ($KIND) from $MODEL"
    if [ "$KIND" = xqc ]; then
        ( export XQC_REPO="$PWD"; set +u; source sota/XQC/pace/env.sh; set -u
          python sota/XQC/capture_xqc_rollout.py --run-dir "$MODEL" --seed-index 0 \
              --env "$VLM_ENV_ID" --out-dir "$OUT" --episodes 2 --max-steps 1000 ) \
        || echo "[capture] $GENE failed"
    else
        env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env \
            python sota/MujocoRL/capture_rollout.py --model "$MODEL" --gene-id "$GENE" \
            --out-dir "$OUT" --env "$VLM_ENV_ID" --episodes 2 --frames 16 --max-steps 1000 \
        || echo "[capture] $GENE failed"
    fi
done < "$VLM_PARENTS_FILE"

# 2. The VLM writes this generation's aux reward (Neel's job script, unchanged).
export AUX_CAPTURE_ROOT="$VLM_CAPTURE_ROOT"
bash aux_reward_observer.sh
