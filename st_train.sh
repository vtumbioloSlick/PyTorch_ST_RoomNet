#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --job-name=image_train
#SBATCH --output=logs/image_train_%j.out
#SBATCH --error=logs/image_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00  # Adjust for long training runs

# === Navigate to your project directory ===
cd $SLURM_SUBMIT_DIR

# === Run your training script ===
python projective_train.py
