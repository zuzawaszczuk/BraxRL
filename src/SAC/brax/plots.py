import matplotlib.pyplot as plt


def make_plot(env_name, xdata, ydata, train_fn):
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

    plt.xlim([0, train_fn.keywords["num_timesteps"]])
    plt.ylim([min_y, max_y])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.plot(xdata, ydata)
    plt.savefig(f"reports/figures/{env_name}_sac_plot_reward.png")


def make_plots(env_name, xdata, metrics_history):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[0, 0].plot(xdata, metrics_history["reward"])
    axs[0, 0].set_title("Episode Reward")

    axs[0, 1].plot(xdata, metrics_history["forward"])
    axs[0, 1].set_title("Forward Reward")

    
    axs[1, 0].plot(xdata, metrics_history["alpha"])
    axs[1, 0].set_title("Alpha (Entropy)")

    axs[1, 1].plot(xdata, metrics_history["actor_loss"])
    axs[1, 1].set_title("Actor Loss")


    axs[2, 0].plot(xdata, metrics_history["critic_loss"])
    axs[2, 0].set_title("Critic Loss")

    axs[2, 1].plot(xdata, metrics_history["reward_std"])
    axs[2, 1].set_title("Reward Std")

    plt.tight_layout()
    plt.savefig(f"reports/figures/{env_name}_sac_full_metrics_grad_up_per_step4.png")
    plt.show()
