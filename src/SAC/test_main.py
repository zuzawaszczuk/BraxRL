from agent import sac_train
from datetime import datetime
from brax import envs

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])
    print("Num steps: {} metrics: {}".format(num_steps, metrics["eval/episode_reward"]))

env = envs.get_environment(env_name="ant", backend="positional")
sac_train(env, progress)
