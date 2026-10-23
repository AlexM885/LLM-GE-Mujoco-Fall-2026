#!/bin/bash
# Submit PPO and/or TD3 seed-config learning curves on HalfCheetah-v4 (3 seeds each),
# to compare with the XQC 10-seed run under the same eval protocol.
#
#   sota/MujocoRL/pace/launch_baseline.sh [ppo] [td3] [--seeds "0 1 2"] [--steps N] [--dry-run]
#
# PPO runs on ice-cpu (no GPU, so it does not count toward the 32 GPU-hour cap).
# TD3 runs on ice-gpu (avoids the broken H100 node on coe-gpu).
# Results: $XQC_RUNS/baseline_{ppo,td3}_halfcheetah_v4/seed*/eval.csv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XQC_REPO="${XQC_REPO:-$(cd "$HERE/../../.." && pwd)}"
JAX_PLATFORMS_USER="${JAX_PLATFORMS-__unset__}"
# shellcheck disable=SC1091
source "$XQC_REPO/sota/XQC/pace/env.sh"
if [ "$JAX_PLATFORMS_USER" = __unset__ ]; then unset JAX_PLATFORMS; fi

ALGOS=(); SEEDS="0 1 2"; STEPS=1000000; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        ppo|td3) ALGOS+=("$1"); shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
done
[ ${#ALGOS[@]} -gt 0 ] || ALGOS=(ppo td3)

# Build the eval venv once here, so parallel jobs don't race to create it.
(cd "$XQC_REPO/sota/MujocoRL" && env -u VIRTUAL_ENV uv sync --frozen --project eval_env)

for algo in "${ALGOS[@]}"; do
    if [ "$algo" = ppo ]; then
        gene=network.py
        res=(-p ice-cpu -c 4 -t 08:00:00)
    else
        gene=seeds/td3_network.py
        res=(-p ice-gpu --gres=gpu:1 -c 4 -t 08:00:00)
    fi
    out="$XQC_RUNS/baseline_${algo}_halfcheetah_v4"
    mkdir -p "$out"
    for seed in $SEEDS; do
        cmd=(sbatch "${res[@]}" --export=ALL,XQC_REPO="$XQC_REPO"
             -J "bl_${algo}_s${seed}" -o "$out/slurm-seed${seed}-%j.out"
             "$HERE/baseline.sbatch" "$algo" "$gene" "$seed" "$STEPS")
        if [ $DRY = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    done
done
