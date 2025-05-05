#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --job-name=cuda_check
#SBATCH --output=logs/cuda_check_%j.out
#SBATCH --error=logs/cuda_check_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00

cd $SLURM_SUBMIT_DIR

python3 test.py
