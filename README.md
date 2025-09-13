# KNSI GOLEM BraxRL Mujoco Repository
[![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml)
[![Pytest](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## What is here?



## Getting started
### Version with UV package manager

When you are using Athena HPC firstly, set up enviroment variables so venvs and cache were in SCRATCH not in HOME. (In home you have limit only 10GB)

```bash
export UV_CACHE_DIR=/net/tscratch/people/{plguser}/.uv/cache
export UV_PROJECT_ENVIRONMENT=/net/tscratch/people/{plguser}/.uv/envs
export UV_LINK_MODE=copy
```

Then download UV package manager with this command.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then go into project folder and run command to download all libraries and dependencies neccesarry for project:
```bash
cd BraxRL
uv sync
```

## Running scripts

Then when you are on Athena you need to prepare run.sh script to run job on GPU.
If you are running localy simple run will be enaugh to create checkpoints, figures, save params and visualizations.

```bash
python src/train_with_visualize.py --env_name "ant"
```

When you are on Athena you need to prepare run.sh script to run job on GPU.

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

# Activate enviroment
source  /net/tscratch/people/{plguser}/.uv/envs/bin/activate

# Navigate to project directory
cd /net/tscratch/people/{plguser}/BraxRL

python utils/test_jax_speed.py
python src/train_with_visualize.py --env_name "ant"
```

