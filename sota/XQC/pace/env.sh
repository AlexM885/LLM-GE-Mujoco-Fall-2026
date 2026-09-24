# Source this on PACE-ICE (login or compute node) before anything XQC related:
#   source <repo>/sota/XQC/pace/env.sh
# Everything (venv, caches, tmp, runs) lives under XQC_BASE in scratch, never in $HOME.

if [ -z "${XQC_BASE:-}" ]; then
    # Default: ~/scratch/xqc-baseline. On PACE-ICE ~/scratch is a symlink into the
    # scratch filesystem; resolve it so nothing ever lands on the home quota.
    if [ -e "$HOME/scratch" ]; then
        XQC_BASE="$(readlink -f "$HOME/scratch")/xqc-baseline"
    else
        echo "env.sh: set XQC_BASE to a directory in your scratch space" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
export XQC_BASE
export XQC_REPO="${XQC_REPO:-$XQC_BASE/repo}"
export XQC_CODE="$XQC_REPO/sota/XQC"
export XQC_VENV="${XQC_VENV:-$XQC_BASE/venv}"
export XQC_RUNS="${XQC_RUNS:-$XQC_BASE/runs}"

# Tools and caches that default to $HOME
export UV_INSTALL_DIR="$XQC_BASE/tools/uv"
export UV_CACHE_DIR="$XQC_BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$XQC_BASE/tools/python"
export PIP_CACHE_DIR="$XQC_BASE/cache/pip"
export XDG_CACHE_HOME="$XQC_BASE/cache/xdg"
export XDG_CONFIG_HOME="$XQC_BASE/cache/xdg-config"
export XDG_DATA_HOME="$XQC_BASE/cache/xdg-data"
export MPLCONFIGDIR="$XQC_BASE/cache/matplotlib"
export WANDB_DIR="$XQC_BASE/cache/wandb"
export WANDB_CACHE_DIR="$XQC_BASE/cache/wandb"
export WANDB_CONFIG_DIR="$XQC_BASE/cache/wandb-config"
export WANDB_DATA_DIR="$XQC_BASE/cache/wandb-data"
export HF_HOME="$XQC_BASE/cache/huggingface"
export CONDA_PKGS_DIRS="$XQC_BASE/cache/conda-pkgs"
export CUDA_CACHE_PATH="$XQC_BASE/cache/nv"
export TMPDIR="${SLURM_TMPDIR:-$XQC_BASE/tmp}"
export PYTHONPYCACHEPREFIX="$XQC_BASE/cache/pycache"

# JAX / MuJoCo runtime
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# jax[cuda12] 0.4.30 errors out (instead of falling back) on nodes without a
# GPU, e.g. login and ice-cpu nodes. Force CPU there.
if [ -z "${JAX_PLATFORMS:-}" ] && ! nvidia-smi -L >/dev/null 2>&1; then
    export JAX_PLATFORMS=cpu
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$XQC_BASE"/{tools,cache,tmp} "$XQC_RUNS" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" \
    "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

export PATH="$UV_INSTALL_DIR:$PATH"
if [ -f "$XQC_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$XQC_VENV/bin/activate"
fi
