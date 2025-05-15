import pickle
from nn import Actor, Critic
from visualization import visualize_policy
from train import train_ddpg
from plots import create_plot
from brax.envs import create

params = {
    "env_name": "ant",
    "episodes": 100,
    "max_steps": 1000,
    "batch_size": 64,
    "save_path": "ddpg_model.pkl",
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "train_episode": 4
}


def save_model(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


env = create(env_name=params["env_name"], backend='generalized')
actor = Actor(action_dim=env.action_size)
critic = Critic()

actor_params, critic_params, episode_rewards = train_ddpg(
    actor,
    critic,
    env,
    episodes=params["train_episode"],
    max_steps=params["max_steps"],
    batch_size=params["batch_size"],
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    tau=params["tau"]
)
create_plot(episode_rewards)
save_model((actor_params, critic_params), params["save_path"])
visualize_policy(actor, actor_params, env)
