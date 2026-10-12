#!/bin/bash
#SBATCH --job-name=limenTrain
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
# One TD3 training of a LIMEN candidate. Same resources as the LLM-GE
# evaluation job (PYTHON_BASH_SCRIPT_TEMPLATE in src/cfg/constants_Mujoco.py).
# Submitted by limen/adapters/gym_mujoco.py through "sbatch --wait"; the
# arguments are the training command.
echo "Launching LIMEN training"
hostname

module load cuda
export CUDA_VISIBLE_DEVICES=0
# The interface functions are tiny; keep JAX off the GPU that TD3 uses
export JAX_PLATFORMS=cpu

"$@"
