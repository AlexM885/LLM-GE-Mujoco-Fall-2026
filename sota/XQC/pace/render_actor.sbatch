#!/bin/bash
#SBATCH --job-name=xqc_render
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:30:00
#SBATCH --output=render-%j.out
# Usage (from sota/XQC): sbatch pace/render_actor.sbatch <gene_id> [<gene_id> ...] [-video_steps 300]
source pace/env.sh   # submit from sota/XQC
cd "$XQC_CODE"
nvidia-smi -L
python render_actor.py "$@"
