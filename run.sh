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
conda activate /net/tscratch/people/plgzwaszczuk/brax_env

# Navigate to project directory
cd /net/tscratch/people/plgzwaszczuk/BraxRL

# Run training
# ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# 
# python src/train.py --env_name "ant"
#python src/train_with_visualize.py --env_name "ant"

for env in ant halfcheetah humanoid humanoidstandup inverted_pendulum inverted_double_pendulum pusher reacher; do
  echo "Train env: $env"
  python src/train_with_visualize.py --env_name "$env"

#   echo "Visualize env: $env"
#   python src/visualize.py --env_name "$env"
#   done