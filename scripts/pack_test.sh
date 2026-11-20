#!/bin/bash
#SBATCH -J packtest
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -G 1
#SBATCH -c 12
#SBATCH --mem 32G
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -o packtest-%j.out
# How many genes fit on one GPU job? Trains N copies of one gene side by side,
# the way run_improved.py packs genes (LLMGE_GENES_PER_JOB), and prints each
# copy's training time, the job's genes per hour, GPU use and peak memory.
#
#   sbatch [-C <gpu type>] [-c cores] [--mem RAM] scripts/pack_test.sh <N> <gene .py file> [timesteps]
#   e.g. sbatch -C V100 -c 16 --mem 64G -t 6:00:00 scripts/pack_test.sh 8 sota/XQC/llmge/network.py
#
# PACK_THREADS=<k> (optional, in the submitting shell) caps each gene's CPU math
# threads (OMP/OpenBLAS/MKL) at k, so N genes don't fight over the job's cores.
#
# Same trainer and settings as a tiered-run gene (3 seeds, 10 eval episodes,
# eval every 25k, shared JAX compile cache). Each copy gets its own gene id,
# because a gene's run dir and results file are named after its id.
set -uo pipefail
if [ $# -lt 2 ]; then
    echo "pack_test.sh got $# argument(s): '$*'. Usage (arguments go AFTER the script name):" >&2
    echo "  sbatch [-C <gpu type>] [-c cores] [--mem RAM] [-t H:MM:SS] scripts/pack_test.sh <N> <gene .py file> [timesteps]" >&2
    exit 1
fi
N=$1; SRC=$2; STEPS=${3:-300000}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f "$SRC" ] || { echo "gene file not found: $SRC (relative to $PWD)" >&2; exit 1; }

export XQC_REPO=$PWD
export XQC_RUNS="$PWD/packtest_runs/${SLURM_JOB_ID:-local}"
source sota/XQC/pace/env.sh
export LLMGE_XQC_EVAL_INTERVAL=25000
export LLMGE_XQC_JAX_CACHE="$XQC_BASE/runs/jax_cache"   # the cache real runs use (warm)
mkdir -p "$XQC_RUNS"
if [ -n "${PACK_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$PACK_THREADS OPENBLAS_NUM_THREADS=$PACK_THREADS MKL_NUM_THREADS=$PACK_THREADS
fi

echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) node=$(hostname) N=$N gene=$SRC steps=$STEPS" \
     "cores=${SLURM_CPUS_PER_TASK:-?} threads_per_gene=${PACK_THREADS:-unlimited}"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "$XQC_RUNS/gpu.csv" &
SMI=$!

START=$(date +%s); PIDS=()
for i in $(seq 1 "$N"); do
    ID="packtest_${SLURM_JOB_ID:-local}_$i"
    cp "$SRC" "sota/XQC/llmge/models/network_$ID.py"
    (
        s=$(date +%s)
        python sota/XQC/train_xqc_gene.py -network "models.network_$ID" -env Walker2d-v5 \
            -timesteps "$STEPS" -num_seeds 3 -seed 0 -eval_episodes 10 > "$XQC_RUNS/$ID.out" 2>&1
        rc=$?
        echo "gene $i: $(( $(date +%s) - s )) s, exit $rc, $(grep -h 'Mean reward' "$XQC_RUNS/$ID.out" | tail -1)"
    ) & PIDS+=($!)
done
wait "${PIDS[@]}"
END=$(date +%s)
kill $SMI 2>/dev/null

SLOWEST=$(( END - START ))
echo "slowest gene (job): $SLOWEST s = $(( SLOWEST / 60 )) min"
awk -v n="$N" -v t="$SLOWEST" 'BEGIN { printf "genes per GPU-hour: %.2f\n", n * 3600 / t }'
awk -F', ' 'NR > 1 { u += $2; if ($3 + 0 > m) m = $3 + 0; k++ } END { if (k) printf "GPU util mean %.0f %%, peak GPU memory %d MiB\n", u / k, m }' "$XQC_RUNS/gpu.csv"
rm -f sota/XQC/llmge/models/network_packtest_${SLURM_JOB_ID:-local}_*.py \
      sota/XQC/llmge/results/packtest_${SLURM_JOB_ID:-local}_*
echo "peak RAM: sacct -j ${SLURM_JOB_ID:-?} -o JobID,MaxRSS,Elapsed  (after the job ends)"
