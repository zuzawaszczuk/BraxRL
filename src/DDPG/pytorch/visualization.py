from brax.io import html
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper

def visualize_policy(actor, env_name, episodes=1, max_steps=1000, device='cuda'):
    env = envs.create(env_name=env_name, backend='spring')
    env = gym_wrapper.VectorGymWrapper(env)
    env = torch_wrapper.TorchWrapper(env, device=device)
    