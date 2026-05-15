import matplotlib.pyplot as plt
import math


def create_plot(step_rewards):
    plt.plot(step_rewards)
    plt.title('DDPG Training Reward')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.savefig('reward_plot.png')

def visualize_obs_history(obs_history):
    n = len(obs_history)
    if n == 0: return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if n == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for i in range(n):
        axs[i].plot(obs_history[i])
        axs[i].set_title(f"Observation {i}")
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig('obs_plots.png')
    plt.close()

def visualize_act_history(act_history):
    n = len(act_history)
    if n == 0: return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if n == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for i in range(n):
        axs[i].plot(act_history[i])
        axs[i].set_title(f"Action {i}")
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig('act_plots.png')
    plt.close()