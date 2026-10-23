#!/bin/bash
# Build the XQC Python environment in scratch. Run on a PACE-ICE login node
# (needs internet), from a clone of the repo:
#   bash sota/XQC/pace/install_env.sh            # GPU build (jax[cuda12])
#   XQC_JAX_EXTRA= bash sota/XQC/pace/install_env.sh   # CPU-only build
# Idempotent: re-running reuses the venv and caches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XQC_REPO="${XQC_REPO:-$(cd "$HERE/../../.." && pwd)}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
JAX_EXTRA="${XQC_JAX_EXTRA-cuda12}"
UV_VERSION="${XQC_UV_VERSION:-0.8.22}"
PY_VERSION="3.9.23"

echo "XQC_BASE=$XQC_BASE"
echo "XQC_REPO=$XQC_REPO"
echo "XQC_VENV=$XQC_VENV"

# 1. uv (single static binary) into scratch
if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv $UV_VERSION into $UV_INSTALL_DIR"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# 2. Python 3.9 (XQC requires exactly 3.9) managed by uv, stored in scratch
uv python install "$PY_VERSION"

# 3. venv in scratch
if [ ! -x "$XQC_VENV/bin/python" ]; then
    uv venv --python "$PY_VERSION" "$XQC_VENV"
fi
# shellcheck disable=SC1091
source "$XQC_VENV/bin/activate"

# 4. pinned deps. jax/jaxlib get the CUDA 12 extra on GPU builds (pip wheels,
#    no CUDA module needed).
REQ="$XQC_CODE/requirements.txt"
if [ -n "$JAX_EXTRA" ]; then
    REQ_TMP="$TMPDIR/xqc-req-$$.txt"
    sed -e "s/^jax==/jax[$JAX_EXTRA]==/" -e "s/^jaxlib==/jaxlib[$JAX_EXTRA]==/" "$REQ" > "$REQ_TMP"
    REQ="$REQ_TMP"
fi
uv pip install -r "$REQ"

# 5. record the exact resolved environment
uv pip freeze > "$XQC_CODE/requirements.lock.txt"
echo "== wrote $XQC_CODE/requirements.lock.txt ($(wc -l < "$XQC_CODE/requirements.lock.txt") packages)"

# 6. quick import check on this node (the real check runs on a compute node:
#    sbatch sota/XQC/pace/check_env.sbatch)
python "$XQC_CODE/pace/check_env.py" --no-gpu-required
