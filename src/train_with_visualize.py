import functools
from datetime import datetime
from brax import envs
from brax.io import model, html
from brax.training.agents.ppo.train import train as ppo_train
# from brax.training.agents.sac.train import train as sac_train
from plots import make_plot
import argparse
import os
import jax
# ml_collections.ConfigDict.to_json = ml_collections.ConfigDict.to_json_best_effort

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='ant')
args = parser.parse_args()

env_name = args.env_name  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']
checkpoint_dir = os.path.abspath(f"reports/checkpoints/{env_name}")

train_fn = {
'inverted_pendulum': functools.partial(ppo_train, num_timesteps=2_000_000, num_evals=20, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1, save_checkpoint_path=checkpoint_dir),
'inverted_double_pendulum': functools.partial(ppo_train, num_timesteps=20_000_000, num_evals=20, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1, save_checkpoint_path=checkpoint_dir),
'ant': functools.partial(ppo_train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1, save_checkpoint_path=checkpoint_dir),
'humanoid': functools.partial(ppo_train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1, save_checkpoint_path=checkpoint_dir),
'reacher': functools.partial(ppo_train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=256, max_devices_per_host=8, seed=1, save_checkpoint_path=checkpoint_dir),
'humanoidstandup': functools.partial(ppo_train, num_timesteps=100_000_000, num_evals=20, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=15, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=6e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1, save_checkpoint_path=checkpoint_dir),
# 'hopper': functools.partial(sac_train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, episode_length=1000, normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=512, grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1, save_checkpoint_path=f"reports/checkpoints/{env_name}"),
# 'walker2d': functools.partial(sac_train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128, grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1, save_checkpoint_path=f"reports/checkpoints/{env_name}"),
'halfcheetah': functools.partial(ppo_train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=0.001, num_envs=2048, batch_size=512, seed=3, save_checkpoint_path=checkpoint_dir),
'pusher': functools.partial(ppo_train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=30, num_minibatches=16, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,entropy_cost=1e-2, num_envs=2048, batch_size=512, seed=3, save_checkpoint_path=checkpoint_dir),
}[env_name]

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    print("Num steps: {} metrics: {}".format(num_steps, metrics["eval/episode_reward"]))


env = envs.get_environment(env_name=env_name, backend=backend)
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

# print(f'time to jit: {times[1] - times[0]}')
# print(f'time to train: {times[-1] - times[1]}')

make_plot(env_name, xdata, ydata, train_fn)

env = envs.create(env_name=env_name, backend=backend)
inference_fn = make_inference_fn(params)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

html_string = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
with open(f"reports/visualizations/{env_name}.html", "w", encoding="utf-8") as f:
    f.write(html_string)
