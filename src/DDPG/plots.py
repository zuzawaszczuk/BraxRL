import matplotlib.pyplot as plt


def create_plot(step_rewards):
    plt.plot(step_rewards)
    plt.title('DDPG Training Reward')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.savefig('reward_plot.png')