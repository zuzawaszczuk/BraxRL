from datetime import datetime
import matplotlib.pyplot as plt
from brax import envs

from agent import sac_train


def make_plot(env_name, xdata, ydata):
    max_y = {
        "ant": 8000,
        "halfcheetah": 8000,
        "hopper": 2500,
        "humanoid": 13000,
        "humanoidstandup": 75_000,
        "reacher": 5,
        "walker2d": 5000,
        "pusher": 0,
    }[env_name]
    min_y = {"reacher": -100, "pusher": -150}.get(env_name, 0)

    # plt.xlim([0, train_fn.keywords["num_timesteps"]])6_553_600
    plt.xlim([0, 150])
    plt.ylim([min_y, max_y])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.plot(xdata, ydata)
    plt.savefig(f"reports/figures/{env_name}_plot_reward_test.png")


env_name = "ant"
xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])
    print("Num steps: {} metrics: {}".format(num_steps, metrics["eval/episode_reward"]))


env = envs.get_environment(env_name, backend="positional")
print(env.observation_size)
sac_train(env, progress)

make_plot(env_name, xdata, ydata)
