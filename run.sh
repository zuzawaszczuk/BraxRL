#!/bin/bash
#SBATCH --job-name=train_braxrl
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml25-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"

# Activate enviroment
conda activate /net/tscratch/people/plguser/brax_env

# Navigate to project directory
cd /net/tscratch/people/plguser

# Run training
python BraxRL/src/train2.py

