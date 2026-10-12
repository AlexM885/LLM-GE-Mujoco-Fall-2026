#!/bin/bash
#SBATCH --job-name=limenTrain
#SBATCH -t 1:30:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
# One TD3 training of a LIMEN candidate (about 30 minutes; the time limit is kept
# short because the cluster caps the GPU-hours of running jobs). Same resources as the LLM-GE
# evaluation job (PYTHON_BASH_SCRIPT_TEMPLATE in src/cfg/constants_Mujoco.py).
# Submitted by limen/adapters/gym_mujoco.py through "sbatch --wait"; the
# arguments are the training command.
echo "Launching LIMEN training"
hostname

module load cuda
# CUDA_VISIBLE_DEVICES is left as SLURM set it: it names the GPU this job was given
# The interface functions are tiny; keep JAX off the GPU that TD3 uses
export JAX_PLATFORMS=cpu

"$@"
