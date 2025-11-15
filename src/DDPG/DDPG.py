import pickle
from nn import Actor, Critic
from visualization import visualize_policy
from train import train_ddpg
from plots import create_plot
from brax.envs import create

params = {
    "env_name": "inverted_pendulum",
    "max_steps": 1000,
    "batch_size": 8,
    "save_path": "ddpg_model.pkl",
    "learning_rate": 1e-2,
    "gamma": 0.95,
    "tau": 0.05,
    "train_episodes": 400,
    "buffer_capacity": 100,
}


def save_model(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


env = create(env_name=params["env_name"])
actor = Actor(action_dim=env.action_size)
critic = Critic()

actor_params, critic_params, episode_rewards = train_ddpg(
    actor,
    critic,
    env_name=params["env_name"],
    episodes=params["train_episodes"],
    max_steps=params["max_steps"],
    batch_size=params["batch_size"],
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    tau=params["tau"],
    buffer_capacity=params["buffer_capacity"],
)
create_plot(episode_rewards)
visualize_policy(actor, actor_params, env, episodes=4, max_steps=500)
