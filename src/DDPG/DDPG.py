import pickle
from nn import Actor, Critic
from visualization import visualize_policy
from plots import create_plot, visualize_obs_history, visualize_act_history
from train import train_ddpg
from brax.envs import get_environment




params = {
    "env_name": "ant",
    "num_timesteps": 100000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_capacity": 2000,
    "exploration_noise": 0.9,
    "start_steps": 5000,
    "terminate_when_unhealthy": True
}

env = get_environment(params["env_name"], terminate_when_unhealthy=params["terminate_when_unhealthy"])
obs_dim = env.observation_size
act_dim = env.action_size

obs_history = [[] for _ in range(obs_dim)]
act_history = [[] for _ in range(act_dim)]
steps = []
episode_rewards = []

def progress_callback(obs, action, reward, step):
    steps.append(step)
    episode_rewards.append(reward)
    for i in range(obs_dim):
        obs_history[i].append(obs[i])
    for i in range(act_dim):
        act_history[i].append(action[i])

actor = Actor(action_dim=env.action_size)
critic = Critic()

actor_params, critic_params, step_rewards, episode_rewards = train_ddpg(
    actor,
    critic,
    env = env,
    num_timesteps=params["num_timesteps"],
    batch_size=params["batch_size"],
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    tau=params["tau"],
    buffer_capacity=params["buffer_capacity"],
    exploration_noise=params["exploration_noise"],
    start_steps=params["start_steps"],
    progress_callback=progress_callback
)
create_plot(step_rewards)
visualize_policy(actor, actor_params, env, episodes=4, max_steps=500)
visualize_obs_history(obs_history)
visualize_act_history(act_history)


