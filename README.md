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
If you want to make these variables persistent, add them to your `.bashrc` file in your HOME directory.

```bash
export UV_CACHE_DIR=/net/tscratch/people/{plguser}/.uv/cache
export UV_PROJECT_ENVIRONMENT=/net/tscratch/people/{plguser}/.uv/envs
export UV_LINK_MODE=copy
```

Then install the UV package manager:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Go into the project folder and install all required libraries and dependencies:
```bash
cd BraxRL
uv sync
```

## Running scripts

If you are running locally, a simple command is enough to create checkpoints, figures, save parameters, and visualizations:

```bash
python src/train_with_visualize.py --env_name "ant"
```

On Athena, you need to prepare a run.sh script to submit a job to the GPU queue:

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
source  /net/tscratch/people/{plguser}/.uv/envs/bin/activate

# Navigate to project directory
cd /net/tscratch/people/{plguser}/BraxRL

python utils/test_jax_speed.py
python src/train_with_visualize.py --env_name "ant"
```

Submit the job with sbatch `run.sh`.

Check the job status with `squeue -u $USER`.

