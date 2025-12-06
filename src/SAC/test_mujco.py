import functools
from datetime import datetime

from brax import envs
from brax.io import model
from brax.training.agents.ppo.train import train as ppo_train

# env_name = (
#     args.env_name
# )  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = "positional"

env = envs.get_environment(env_name="ant", backend=backend)
# env = brax.envs.create()
print(env.action_size)  # -> 8
print(env.observation_size)
