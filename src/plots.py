import matplotlib.pyplot as plt

max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'pusher': 0}[env_name]
min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)

plt.xlim([0, train_fn.keywords['num_timesteps']])
plt.ylim([min_y, max_y])
plt.xlabel('# environment steps')
plt.ylabel('reward per episode')
plt.plot(xdata, ydata)
plt.savefig(f"reports/figures/{env_name}_plot_reward.png")