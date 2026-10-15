#!/usr/bin/env bash
# Train the baseline seed gene once and capture rollouts for the VLM.
#
# Writes a captured parent (frames + telemetry + manifest) to
# $VLM_PILOT_ROOT/captures/vlm_pilot_seed/. The VLM aux_reward emitter in
# the next step reads that capture via src/visual_feedback.load_capture.
#
# Env in:
#   VLM_PILOT_ROOT   shared scratch root for this pilot
#   PILOT_TIMESTEPS  training budget for the capture gene (default 250000)
#   CAPTURE_EPISODES rollouts to capture (default 2)
#   CAPTURE_FRAMES   frames per rollout (default 32)
set -euo pipefail
: "${VLM_PILOT_ROOT:?set VLM_PILOT_ROOT first}"
PILOT_TIMESTEPS="${PILOT_TIMESTEPS:-250000}"
CAPTURE_EPISODES="${CAPTURE_EPISODES:-2}"
CAPTURE_FRAMES="${CAPTURE_FRAMES:-32}"
GENE_ID="vlm_pilot_seed"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

mkdir -p "$VLM_PILOT_ROOT/stats" "$VLM_PILOT_ROOT/trained_models" \
         "$VLM_PILOT_ROOT/captures" "$VLM_PILOT_ROOT/results"
mkdir -p sota/MujocoRL/models run_job_outputs/vlm_pilot
touch sota/MujocoRL/models/__init__.py
cp -f sota/MujocoRL/network.py "sota/MujocoRL/models/network_${GENE_ID}.py"

echo "[capture] training the seed gene for $PILOT_TIMESTEPS steps"
JOB_TRAIN=$(sbatch --parsable \
    --job-name=vlm_pilot_capture_train \
    -p ice-gpu \
    -t 2:00:00 --gres=gpu:1 --mem 32G -c 8 \
    --output="$VLM_PILOT_ROOT/slurm_capture_train_%j.out" \
    --wrap="set -e; module load cuda; module load uv; \
            export UV_CACHE_DIR=\$HOME/.cache/uv; export PYTHONUNBUFFERED=1; \
            cd $ROOT/sota/MujocoRL && \
            env -u VIRTUAL_ENV uv run --isolated --project eval_env python train_rl.py \
                -network models.network_${GENE_ID} \
                -timesteps $PILOT_TIMESTEPS \
                -model_dir $VLM_PILOT_ROOT/trained_models \
                -stats_dir $VLM_PILOT_ROOT/stats")
echo "[capture] train job id: $JOB_TRAIN"

echo "[capture] capturing $CAPTURE_EPISODES rollouts after training"
JOB_CAPTURE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_TRAIN" \
    --job-name=vlm_pilot_capture_rollout \
    -p ice-gpu \
    -t 0:30:00 --gres=gpu:1 --mem 16G -c 4 \
    --output="$VLM_PILOT_ROOT/slurm_capture_rollout_%j.out" \
    --wrap="set -e; module load cuda; module load uv; \
            export UV_CACHE_DIR=\$HOME/.cache/uv; export PYTHONUNBUFFERED=1; \
            cd $ROOT && \
            env -u VIRTUAL_ENV uv run --isolated --project sota/MujocoRL/eval_env python sota/MujocoRL/capture_rollout.py \
                --model $VLM_PILOT_ROOT/trained_models/${GENE_ID}.zip \
                --out-dir $VLM_PILOT_ROOT/captures/${GENE_ID} \
                --gene-id ${GENE_ID} \
                --genome sota/MujocoRL/models/network_${GENE_ID}.py \
                --episodes $CAPTURE_EPISODES \
                --frames $CAPTURE_FRAMES \
                --max-steps 1000")
echo "[capture] capture job id: $JOB_CAPTURE"
echo "[capture] once both complete, run 01_emit_aux_reward.sh"
# Print job ids one more time to a known file so later agent turns recover state
echo "{\"train_job\": $JOB_TRAIN, \"capture_job\": $JOB_CAPTURE, \"gene_id\": \"$GENE_ID\"}" \
  > "$VLM_PILOT_ROOT/jobs_00.json"
