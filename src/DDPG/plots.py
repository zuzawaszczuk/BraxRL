import matplotlib.pyplot as plt


def create_plot(episode_rewards):
    plt.plot(episode_rewards)
    plt.title('DDPG Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('reward_plot.png')