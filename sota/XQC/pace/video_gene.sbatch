#!/bin/bash
#SBATCH --job-name=xqc_video
#SBATCH -p ice-gpu
#SBATCH -C nvidia-gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH --output=video-%j.out
# Usage (from sota/XQC): sbatch pace/video_gene.sbatch llmge/models/network_<id>.py [-name NAME] [...]
source pace/env.sh   # submit from sota/XQC
cd "$XQC_CODE"
nvidia-smi -L
python video_gene.py "$@"
