#!/bin/bash
#SBATCH --job-name=mesb-llm-server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=mujoco_rl_output/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Start the project's local LLM server (server.py, MODEL_PATH in
# src/cfg/constants_Mujoco.py) for a MESB run. Same resources as server.sh,
# with two differences:
#   * it listens on PORT from src/cfg/constants_Mujoco.py (server.sh at this
#     commit hard-codes 8137 while the constants say 8169; see docs/MESB.md);
#   * it never submits island_controller.sbatch.
# 8 hours is the longest a 2-GPU job may run under the GPU QOS (a longer
# request waits forever with QOSMaxGRESMinutesPerJob). For longer runs use
# scripts/mesb/start_llm_server.sh N to queue N servers back to back.
# It writes the node name to hostname.log, which the LLM operators read.
#
#   mkdir -p mujoco_rl_output/slurm_logs && sbatch scripts/mesb/llm_server.sbatch
#   scripts/mesb/start_llm_server.sh 3        # 3 x 8 h, back to back
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
    module load uv 2>/dev/null || true
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"

PORT="$(uv run python -c 'import sys; sys.path.insert(0, "src"); from cfg import constants; print(constants.PORT)')"
SERVER_HOSTNAME="$(hostname)"
export SERVER_HOSTNAME
echo "$SERVER_HOSTNAME" > hostname.log
echo "LLM server for MESB on ${SERVER_HOSTNAME}:${PORT} (hostname.log written)"

uv run python -m uvicorn server:app --host "$SERVER_HOSTNAME" --port "$PORT" --workers 1
