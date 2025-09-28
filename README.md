# KNSI GOLEM BraxRL Mujoco Repository

[![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml)
[![Pytest](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## What is here?

BraxRL is a lightweight research project focused on implementing and experimenting with reinforcement learning algorithms in JAX.
It includes clean implementations of SAC, TD3, and DDPG, and tests them on continuous control tasks using the MuJoCo physics engine via Brax.
The goal is to provide a codebase for learning and benchmarking modern RL algorithms on GPU-accelerated environments.

## Getting started
### Version with UV Package Manager

When using the Athena HPC, first set up environment variables so that virtual environments and cache are stored in **SCRATCH**, not in **HOME** (HOME has a 10 GB limit).  
If you want to prepare local environment go to header **UV manager installation**, which is below.

```bash
# Setup .bashrc file adding these lines there on Athena HPC
export UV_CACHE_DIR=/net/tscratch/people/{plguser}/.uv/cache
export UV_PROJECT_ENVIRONMENT=/net/tscratch/people/{plguser}/.uv/envs
export UV_LINK_MODE=copy

# UV manager installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# All required libraries and dependencies installation in project directory:
cd BraxRL
uv sync
```

## Available Enviroments

 - ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']. 

When you choose one enviroment, substitude e.g. {ENV_NAME} = ant in further sections.

## Running scripts - only them and with SLURM configuration 

```bash
# If you are running only scripts, should activate firstly venv environment:
source /net/tscratch/people/{plguser}/.uv/envs/bin/activate

# Then, you may type line in order to launch ant's training by uv manager  
uv run src/train_with_visualize.py --env_name "{ENV_NAME}" 

# Or with usual interpreter
python src/train_with_visualize.py --env_name "{ENV_NAME}" 
```

In order to launch {ENV_NAME} environment, go to reports/visualizations directory and launch {ENV_NAME}.html format in browser.

* On Athena, you need to prepare a run.sh script to submit a job to the GPU queue:

```bash
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
python src/train_with_visualize.py --env_name "{ENV_NAME}"

# Then, give an access to file and launch it easily:
chmod +x run.sh
./run.sh
```

###  Check the job status in order to look on resource queue typing:
```bash
squeue -u $USER
```