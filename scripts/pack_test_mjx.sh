#!/bin/bash
#SBATCH -J packtest_mjx
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -G V100:1
#SBATCH -c 12
#SBATCH --mem 32G
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -o packtest_mjx-%j.out
# pack_test.sh with MJX instead of CPU MuJoCo: trains N copies of one gene side
# by side on one GPU with sota/XQC/train_xqc_gene_mjx.py (same XQC, same budget,
# Walker2d-v5 simulated by MJX, whole training loop compiled on the GPU) and
# prints each copy's time, the job's genes per hour, GPU use and peak memory.
#
#   sbatch [-G <gpu type>:1] scripts/pack_test_mjx.sh <N> <gene .py file> [timesteps]
#   e.g. sbatch scripts/pack_test_mjx.sh 3 sota/XQC/llmge/network.py   (V100 by default)
#        sbatch -G H200:1 scripts/pack_test_mjx.sh 3 sota/XQC/llmge/network.py
#
# Same settings as pack_test.sh (3 seeds, 10 eval episodes, shared JAX compile
# cache). There are no evaluations during training (pack_test.sh's every-25k
# learning curve), so compare the two on genes per hour, not on curves.
# MJX_SOLVER=fast caps MJX's constraint solver iterations (default: the XML's).
set -uo pipefail
if [ $# -lt 2 ]; then
    echo "pack_test_mjx.sh got $# argument(s): '$*'. Usage (arguments go AFTER the script name):" >&2
    echo "  sbatch [-G <gpu type>:1] [-t H:MM:SS] scripts/pack_test_mjx.sh <N> <gene .py file> [timesteps]" >&2
    exit 1
fi
N=$1; SRC=$2; STEPS=${3:-300000}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f "$SRC" ] || { echo "gene file not found: $SRC (relative to $PWD)" >&2; exit 1; }

export XQC_REPO=$PWD
export XQC_RUNS="$PWD/packtest_runs/${SLURM_JOB_ID:-local}"
source sota/XQC/pace/env.sh
export LLMGE_XQC_JAX_CACHE="$XQC_BASE/runs/jax_cache_mjx"
export LLMGE_MJX_SOLVER="${MJX_SOLVER:-default}"
mkdir -p "$XQC_RUNS"

# mujoco-mjx (matching the venv's mujoco 3.1.6) and trimesh go in a side
# directory; the venv itself is left as is. Everything else MJX needs is there.
MJX_SITE="$XQC_BASE/mjx_site"
export PYTHONPATH="$MJX_SITE${PYTHONPATH:+:$PYTHONPATH}"
if ! python -c "import mujoco.mjx" 2>/dev/null; then
    echo "installing mujoco-mjx into $MJX_SITE"
    MJ=$(python -c "import mujoco; print(mujoco.__version__)")
    uv pip install --no-deps --target "$MJX_SITE" "mujoco-mjx==$MJ" trimesh \
        || python -m pip install --no-deps --target "$MJX_SITE" "mujoco-mjx==$MJ" trimesh
    python -c "import mujoco.mjx" || { echo "mujoco.mjx still does not import" >&2; exit 1; }
fi

echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) node=$(hostname) N=$N gene=$SRC steps=$STEPS solver=$LLMGE_MJX_SOLVER"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "$XQC_RUNS/gpu.csv" &
SMI=$!

START=$(date +%s); PIDS=()
for i in $(seq 1 "$N"); do
    ID="packtest_mjx_${SLURM_JOB_ID:-local}_$i"
    cp "$SRC" "sota/XQC/llmge/models/network_$ID.py"
    (
        s=$(date +%s)
        python sota/XQC/train_xqc_gene_mjx.py -network "models.network_$ID" -env Walker2d-v5 \
            -timesteps "$STEPS" -num_seeds 3 -seed 0 -eval_episodes 10 > "$XQC_RUNS/$ID.out" 2>&1
        rc=$?
        echo "gene $i: $(( $(date +%s) - s )) s, exit $rc, $(grep -h '\[timing\]' "$XQC_RUNS/$ID.out" | tail -1)"
        echo "        $(grep -h 'Mean reward' "$XQC_RUNS/$ID.out" | tail -1)"
    ) & PIDS+=($!)
done
wait "${PIDS[@]}"
END=$(date +%s)
kill $SMI 2>/dev/null

SLOWEST=$(( END - START ))
echo "slowest gene (job): $SLOWEST s = $(( SLOWEST / 60 )) min"
awk -v n="$N" -v t="$SLOWEST" 'BEGIN { printf "genes per GPU-hour: %.2f\n", n * 3600 / t }'
awk -F', ' 'NR > 1 { u += $2; if ($3 + 0 > m) m = $3 + 0; k++ } END { if (k) printf "GPU util mean %.0f %%, peak GPU memory %d MiB\n", u / k, m }' "$XQC_RUNS/gpu.csv"
rm -f sota/XQC/llmge/models/network_packtest_mjx_${SLURM_JOB_ID:-local}_*.py \
      sota/XQC/llmge/results/packtest_mjx_${SLURM_JOB_ID:-local}_*
echo "peak RAM: sacct -j ${SLURM_JOB_ID:-?} -o JobID,MaxRSS,Elapsed  (after the job ends)"
