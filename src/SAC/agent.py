import os
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from actor import ActorNetwork, sample_normal
from brax import envs
from critic import CriticNetwork
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints, train_state
from jaxtyping import Array, Float, PRNGKeyArray
from value import ValueNetwork


@dataclass
class TrainStates:
    actor_state: train_state.TrainState
    critic1_state: train_state.TrainState
    critic2_state: train_state.TrainState
    value_state: train_state.TrainState
    target_value_state: train_state.TrainState


def sac_train(
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
    save_checkpoint_path=f"reports/checkpoints/ant",
):
    buffer = fbx.make_flat_buffer(
        max_length=max_replay_size,
        min_length=min_replay_size,
        sample_batch_size=batch_size,
    )

    env = envs.create("ant")
    max_action = 1

    train_states = init_train_states(env, learning_rate)

    print("sukces")


def choose_action(
    apply: Callable[[FrozenDict, Array], tuple[Array, Array]],
    actor_params: FrozenDict,
    state: Array,
    max_action: float,
    key: PRNGKeyArray,
    reparam_noise: float = 1e-6,
):
    actions, _ = sample_normal(
        apply, actor_params, state, max_action, key, reparam_noise, reparameterize=False
    )

    return jnp.squeeze(actions, axis=0)


example_timestep = {
    "state": jnp.zeros((10,)),
    "action": jnp.zeros((5,)),
    "reward": jnp.array(0.0),
    "next_state": jnp.zeros((10,)),
    "done": jnp.array(False),
}
example_second = {
    "state": jnp.zeros((10,)),
    "action": jnp.zeros((5,)),
    "reward": jnp.array(0.0),
    "next_state": jnp.zeros((10,)),
    "done": jnp.array(False),
}

# buffer_state = buffer.init(example_timestep)
# buffer_state = buffer.add(buffer_state, example_second)
# batch = buffer.sample(buffer_state, rng)


# def init_train_states(
#     env: envs.Env,
#     learning_rate: float = 3e-4,
#     rng: PRNGKeyArray = jax.random.PRNGKey(0),
# ) -> TrainStates:
#     rng, *keys = jax.random.split(rng, 6)
#     actor_key, critic1_key, critic2_key, value_key, target_value_key = keys

#     actor = ActorNetwork(env.observation_size, env.action_size)
#     critic1 = CriticNetwork(env.observation_size, env.action_size)
#     critic2 = CriticNetwork(env.observation_size, env.action_size)
#     value = ValueNetwork(env.observation_size)
#     target_value = ValueNetwork(env.observation_size)

#     dummy_obs = jnp.zeros((1, env.observation_size))
#     dummy_action = jnp.zeros((1, env.action_size))

#     actor_params = actor.init(actor_key, dummy_obs)
#     critic1_params = critic1.init(critic1_key, dummy_obs, dummy_action)
#     critic2_params = critic2.init(critic2_key, dummy_obs, dummy_action)
#     value_params = value.init(value_key, dummy_obs)
#     target_value_params = target_value.init(target_value_key, dummy_obs)

#     actor_state = train_state.TrainState.create(
#         apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate)
#     )
#     critic1_state = train_state.TrainState.create(
#         apply_fn=critic1.apply, params=critic1_params, tx=optax.adam(learning_rate)
#     )
#     critic2_state = train_state.TrainState.create(
#         apply_fn=critic2.apply, params=critic2_params, tx=optax.adam(learning_rate)
#     )
#     value_state = train_state.TrainState.create(
#         apply_fn=value.apply, params=value_params, tx=optax.adam(learning_rate)
#     )
#     target_value_state = train_state.TrainState.create(
#         apply_fn=target_value.apply,
#         params=target_value_params,
#         tx=optax.adam(learning_rate),
#     )

#     return TrainStates(
#         actor_state, critic1_state, critic2_state, value_state, target_value_state
#     )


def save_checkpoints(train_states: TrainStates, step: int, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states.actor_state, step=step, prefix="actor_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states.critic1_state, step=step, prefix="critic1_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states.critic2_state, step=step, prefix="critic2_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir, train_states.value_state, step=step, prefix="value_"
    )
    checkpoints.save_checkpoint(
        checkpoint_dir,
        train_states.target_value_state,
        step=step,
        prefix="target_value_",
    )


def restore_checkpoints(train_states: TrainStates, checkpoint_dir: str) -> TrainStates:
    actor_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states.actor_state, prefix="actor_"
    )
    critic1_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states.critic1_state, prefix="critic1_"
    )
    critic2_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states.critic2_state, prefix="critic2_"
    )
    value_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states.value_state, prefix="value_"
    )
    target_value_state = checkpoints.restore_checkpoint(
        checkpoint_dir, train_states.target_value_state, prefix="target_value_"
    )
    return TrainStates(
        actor_state, critic1_state, critic2_state, value_state, target_value_state
    )
