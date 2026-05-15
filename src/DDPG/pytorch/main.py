from train import train

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
}

xdata = []
ydata = []
def progress_callback(t, episode_rewards):
    xdata.append(t)
    ydata.append(episode_rewards[-1])
    print(f"Step: {t} | Episode Reward: {episode_rewards[-1]:.2f}")

agent = train(
    env_name=params["env_name"],
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

