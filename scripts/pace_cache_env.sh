#!/bin/bash
# Route job-local package/model caches away from shared read-mostly model dirs.
# Source this from Slurm scripts before invoking uv, transformers, torch, or HF Hub.

LLMGE_CACHE_ROOT="${LLMGE_CACHE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}/.cache}"
LLMGE_SHARED_HF_HOME="${LLMGE_SHARED_HF_HOME:-/storage/ice-shared/vip-vvk/llm_storage}"
mkdir -p "$LLMGE_CACHE_ROOT"/{uv,huggingface,torch,xdg}

export UV_CACHE_DIR="${UV_CACHE_DIR:-$LLMGE_CACHE_ROOT/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LLMGE_CACHE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$LLMGE_CACHE_ROOT/torch}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ "${LLMGE_USE_SCRATCH_HF:-0}" = "1" ]; then
    export HF_HOME="${HF_HOME:-$LLMGE_CACHE_ROOT/huggingface}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
else
    export HF_HOME="${HF_HOME:-$LLMGE_SHARED_HF_HOME}"
fi

mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
if [ "${LLMGE_USE_SCRATCH_HF:-0}" = "1" ]; then
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE"
fi

echo "Using cache root: $LLMGE_CACHE_ROOT"
echo "  UV_CACHE_DIR=$UV_CACHE_DIR"
echo "  HF_HOME=$HF_HOME"
if [ "${LLMGE_USE_SCRATCH_HF:-0}" = "1" ]; then
    echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
    echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
    echo "  HF_XET_CACHE=$HF_XET_CACHE"
fi
echo "  TORCH_HOME=$TORCH_HOME"
