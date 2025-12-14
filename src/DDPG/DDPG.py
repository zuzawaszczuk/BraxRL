import pickle
from nn import Actor, Critic
from visualization import visualize_policy
from train import train_ddpg
from plots import create_plot
from brax.envs import create

params = {
    "env_name": "ant",
    "num_timesteps": 1000000,
    "batch_size": 32,
    "learning_rate": 1e-2,
    "gamma": 0.95,
    "tau": 0.05,
    "buffer_capacity": 1000,
}


env = create(env_name=params["env_name"])
actor = Actor(action_dim=env.action_size)
critic = Critic()

actor_params, critic_params, step_rewards, episode_rewards = train_ddpg(
    actor,
    critic,
    env_name=params["env_name"],
    num_timesteps=params["num_timesteps"],
    batch_size=params["batch_size"],
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    tau=params["tau"],
    buffer_capacity=params["buffer_capacity"],
)
create_plot(step_rewards)
visualize_policy(actor, actor_params, env, episodes=4, max_steps=500)
