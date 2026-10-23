#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
echo "launching LLM Guided Evolution"
hostname
module load uv
module load cuda

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export LLMGE_PORT="${LLMGE_PORT:-8169}"

# Run options (robot, seed, search method): src/cfg/run_options.py, each one
# overridable from the environment, e.g.
#   LLMGE_ROBOT=halfcheetah LLMGE_SEED=xqc LLMGE_RUN=xqc_cheetah sbatch run.sh
#   USE_LLM_EMITTERS=0 LLMGE_SLIDING_BOUNDARIES=1 sbatch run.sh
# sbatch passes this environment on to the LLM and evaluation jobs, so every
# process of the run reads the same options. Combinations that cannot work
# together stop here, before anything is submitted.
OPTIONS=$(python3 -c 'import sys; sys.path.insert(0, "src/cfg"); import run_options as o
print(o.SEED, int(o.SLIDING_BOUNDARIES)); print(o.describe_run_options(), file=sys.stderr)') || exit 1
read -r SEED_OPTION SLIDING <<< "$OPTIONS"
RUN="${LLMGE_RUN:-mujoco_islands_run}"

if [ "$SLIDING" = 1 ]; then
    # Sliding boundaries run their own driver, run_mesb.py; its job script reads
    # the same run options (MESB_* variables tune it, see the script).
    echo "Sliding boundaries on: running run_mesb.py through scripts/mesb/run_mesb.sbatch"
    RUN_NAME="${RUN_NAME:-$RUN}" bash scripts/mesb/run_mesb.sbatch
    exit $?
fi

if [ "$SEED_OPTION" = xqc ]; then PROMPT_GROUP=Mujoco/XQC; else PROMPT_GROUP=Mujoco/Normal; fi
echo "run=$RUN prompts=$PROMPT_GROUP pop=${LLMGE_POP_SIZE:-32}"

uv run python run_improved.py \
	--checkpoints "$RUN/island_llama3_${PROMPT_GROUP//\//-}" \
	--global_path "$RUN/global_data" \
	--llm_model llama3 \
	--prompt_group "$PROMPT_GROUP"
