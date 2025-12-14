import os
from dataclasses import dataclass
from typing import TypeAlias

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.envs.base import State as EnvState
from flashbax.buffers.trajectory_buffer import (Experience, TrajectoryBuffer,
                                                TrajectoryBufferState)
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import jit
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray

from actor import ActorNetwork, sample_normal
from critic import CriticNetwork
from training import learn, create_train_states, TrainStates
from value import ValueNetwork

BufferState: TypeAlias = TrajectoryBufferState[Experience]


def sac_train(
    env: envs,
    progress_fn,
    num_timesteps=6_553_600,
    num_evals=20,
    reward_scaling=30,  #
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    discounting=0.997,
    learning_rate=6e-4,
    num_envs=128,
    batch_size=512,  #
    grad_updates_per_step=64,
    max_devices_per_host=1,
    max_replay_size=1048576,  #
    min_replay_size=8192,
    seed=1,
    save_checkpoint_path="reports/checkpoints/ant",
):
    buffer = fbx.make_flat_buffer(
        max_length=max_replay_size,
        min_length=min_replay_size,
        sample_batch_size=batch_size,
    )

    max_action = 1
    rng = jax.random.PRNGKey(seed)
    buffer_state = init_buffer(buffer, env.action_size, env.observation_size)
    params = init_train_states(env, learning_rate)

    buffer_state, rng = collect_start_samples(
        buffer, buffer_state, env, params, max_action, min_replay_size, rng
    )
    print("collected")

    time_steps = min_replay_size

    while time_steps < num_timesteps:
        rng, rng1 = jax.random.split(rng)
        observation = env.reset(rng1)
        done = False
        score = 0
        episode_steps = 0

        while not done and episode_steps < episode_length and time_steps < num_timesteps:
            rng, rng_action = jax.random.split(rng)
            action = choose_action(
                params["actor"], observation.obs, max_action, rng_action
            )

            observation_ = env.step(observation, action)
            score += observation_.reward
            done = observation_.done

            remember = create_buffer_state(observation, observation_, action)
            buffer_state = buffer.add(buffer_state, remember)
            rng, rng_learn = jax.random.split(rng)
            params, rng = learn(
                buffer,
                buffer_state,
                params,
                max_action,
                rng_learn,
                reward_scaling,
                discounting,
                grad_updates_per_step,
            )
            if time_steps % 10 == 0:
                metrics = {"eval/episode_reward": score}
                progress_fn(time_steps, metrics)
            episode_steps += 1
            time_steps += 1

        print(f"Timestep score: {score}")


def choose_action(
    actor: TrainState,
    state: Array,
    max_action: float,
    key: PRNGKeyArray,
    reparam_noise: float = 1e-6,
) -> Array:
    actions, _ = sample_normal(actor, actor.params, state, max_action, key, False, reparam_noise)

    return jnp.squeeze(actions, axis=0)


def init_buffer(
    buffer: TrajectoryBuffer, action_size: int, obs_size: int
) -> BufferState:
    example_timestep = {
        "state": jnp.zeros((obs_size,)),
        "action": jnp.zeros((action_size,)),
        "reward": jnp.array(0.0),
        "next_state": jnp.zeros((obs_size,)),
        "done": jnp.array(False),
    }
    return buffer.init(example_timestep)


def create_buffer_state(
    observation: EnvState, observation_: EnvState, action: Array
) -> BufferState:
    return {
        "state": observation.obs,
        "action": action,
        "reward": observation_.reward,
        "next_state": observation_.obs,
        "done": jnp.asarray(observation_.done, dtype=jnp.bool_),
    }


def collect_start_samples(
    buffer: TrajectoryBuffer,
    buffer_state: BufferState,
    env: envs,
    params: TrainStates,
    max_action: int,
    min_replay_size: int,
    key: PRNGKeyArray,
) -> tuple[BufferState, PRNGKeyArray]:
    observation = env.reset(key)

    def one_step(carry, _):
        observation, buffer_state, key = carry
        key, subkey = jax.random.split(key)
        action = choose_action(params["actor"], observation.obs, max_action, subkey)
        observation_ = env.step(observation, action)
        remember = create_buffer_state(observation, observation_, action)
        buffer_state = buffer.add(buffer_state, remember)
        return (observation_, buffer_state, key), None

    (_, buffer_state, key), _ = scan(
        jit(one_step), (observation, buffer_state, key), None, length=min_replay_size
    )

    return buffer_state, key


def init_train_states(
    env: envs.Env,
    learning_rate: float = 3e-4,
    rng: PRNGKeyArray = jax.random.PRNGKey(0),
) -> TrainStates:
    rng, *keys = jax.random.split(rng, 6)
    actor_key, critic1_key, critic2_key, value_key, target_value_key = keys

    actor = ActorNetwork(env.observation_size, env.action_size)
    critic1 = CriticNetwork(env.observation_size, env.action_size)
    critic2 = CriticNetwork(env.observation_size, env.action_size)
    value = ValueNetwork(env.observation_size)
    target_value = ValueNetwork(env.observation_size)

    dummy_obs = jnp.zeros((1, env.observation_size))
    dummy_action = jnp.zeros((1, env.action_size))

    actor_params = actor.init(actor_key, dummy_obs)
    critic1_params = critic1.init(critic1_key, dummy_obs, dummy_action)
    critic2_params = critic2.init(critic2_key, dummy_obs, dummy_action)
    value_params = value.init(value_key, dummy_obs)
    target_value_params = target_value.init(target_value_key, dummy_obs)

    actor_state = TrainState.create(
        apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate)
    )
    critic1_state = TrainState.create(
        apply_fn=critic1.apply, params=critic1_params, tx=optax.adam(learning_rate)
    )
    critic2_state = TrainState.create(
        apply_fn=critic2.apply, params=critic2_params, tx=optax.adam(learning_rate)
    )
    value_state = TrainState.create(
        apply_fn=value.apply, params=value_params, tx=optax.adam(learning_rate)
    )
    target_value_state = TrainState.create(
        apply_fn=target_value.apply,
        params=target_value_params,
        tx=optax.adam(learning_rate),
    )

    return create_train_states(
        actor_state, critic1_state, critic2_state, value_state, target_value_state
    )


def save_checkpoints(train_states: TrainStates, step: int, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states["actor"], step=step, prefix="actor_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states["critic1"], step=step, prefix="critic1_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states["critic2"], step=step, prefix="critic2_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states["value"], step=step, prefix="value_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir,
        train_states["target_value"],
        step=step,
        prefix="target_value_",
    )


def restore_checkpoints(train_states: TrainStates, checkpoint_dir: str) -> TrainStates:
    actor_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states["actor"], prefix="actor"
    )
    critic1_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states["critic1"], prefix="critic1"
    )
    critic2_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states["critic2"], prefix="critic2"
    )
    value_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states["value"], prefix="value"
    )
    target_value_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states["target_value"], prefix="target_value"
    )
    return TrainStates(
        actor_state, critic1_state, critic2_state, value_state, target_value_state
    )
