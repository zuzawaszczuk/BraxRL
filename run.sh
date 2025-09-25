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

# Activate environment
source /net/tscratch/people/{plguser}/.uv/envs/bin/activate 

# Navigate to project directory
cd /net/tscratch/people/{plguser}/BraxRL

### {plguser} = your current login

python utils/test_jax_speed.py
python src/train_with_visualize.py --env_name "ant"

# Run training
# ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# 
# python src/train.py --env_name "ant"




