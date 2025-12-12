from dataclasses import dataclass
from typing import Tuple, TypeAlias

import jax
import jax.numpy as jnp
from actor import sample_normal
from flashbax.buffers.trajectory_buffer import (Experience, TrajectoryBuffer,
                                                TrajectoryBufferSample,
                                                TrajectoryBufferState)
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray

BufferState: TypeAlias = TrajectoryBufferState[Experience]


@dataclass
class TrainStates:
    actor: TrainState
    critic1: TrainState
    critic2: TrainState
    value: TrainState
    target_value: TrainState


def learn(
    buffer: TrajectoryBuffer,
    buffer_state: BufferState,
    params: TrainStates,
    max_action: float,
    key: PRNGKeyArray,
    reward_scaling=30,
    discounting=0.997,
) -> TrainStates | None:
    if not buffer.can_sample(buffer_state):
        return None

    key, *keys = jax.random.split(key, 3)
    buffer_key, actor_key1, actor_key2 = keys

    batch = buffer.sample(buffer_state, buffer_key)

    new_value = update_value_network(params, batch, max_action, actor_key1)
    new_target_value = soft_update_target_value_network(params.target_value, new_value)
    new_actor = update_actor_network(params, batch, max_action, actor_key2)
    new_critic1, new_critic2 = update_critic_networks(
        params, batch, reward_scaling, discounting
    )
    print(f" actor {new_actor} critic1 {new_critic1} ")

    return TrainStates(new_actor, new_critic1, new_critic2, new_value, new_target_value)


@jax.jit
def update_value_network(
    params: TrainStates,
    batch: TrajectoryBufferSample[Experience],
    max_action: float,
    key: PRNGKeyArray,
) -> TrainState:
    actions, log_probs = sample_normal(
        params.actor, batch["state"], max_action, key, False
    )

    q1_new_policy = params.critic1.apply_fn(params.critic1, batch["state"], actions)
    q2_new_policy = params.critic2.apply_fn(params.critic2, batch["state"], actions)

    critic_value = jnp.minimum(q1_new_policy, q2_new_policy)
    # y = min(Q1, Q2) - log π(a|s)
    value_target = critic_value - log_probs  # alpha = 1

    # L_v = 1/2 * mean( (V(s) - y)^2
    def value_loss_fn(value_params):
        v = value_params.apply_fn(value_params.params, batch["state"])
        loss = 0.5 * jnp.mean((v - value_target) ** 2)
        return loss

    grads = jax.grad(value_loss_fn)(params.value)
    new_value = params.value.apply_gradients(grads=grads)

    return new_value


@jax.jit
def soft_update_target_value_network(
    target_value: TrainState, new_value: TrainState, tau: float = 0.05
) -> TrainState:
    new_target_value_params = jax.tree_util.tree_map(
        lambda target, main: (1 - tau) * target + tau * main,
        target_value.params,
        new_value.params,
    )
    return target_value.replace(params=new_target_value_params)


@jax.jit
def update_actor_network(
    params: TrainStates,
    batch: TrajectoryBufferSample[Experience],
    max_action: float,
    key: PRNGKeyArray,
) -> TrainState:

    def actor_loss_fn(actor_params):
        actions, log_probs = sample_normal(
            actor_params, batch["state"], max_action, key, reparameterize=True
        )

        q1 = params.critic1.apply_fn(params.critic1.params, batch["state"], actions)
        q2 = params.critic2.apply_fn(params.critic2.params, batch["state"], actions)
        q_min = jnp.minimum(q1, q2)

        # L_actor = α log π(a|s) - Q(s,a), α=1 for simplicity
        loss = jnp.mean(log_probs - q_min)
        return loss

    grads = jax.grad(actor_loss_fn)(params.actor.params)
    new_actor = params.actor.apply_gradients(grads=grads)

    return new_actor


@jax.jit
def update_critic_networks(
    params: TrainStates,
    batch: TrajectoryBufferSample[Experience],
    reward_scaling: float = 30,
    gamma: float = 0.99,
) -> Tuple[TrainState, TrainState]:
    value_ = params.target_value.apply_fn(params.target_value.params, batch["state"])
    q_hat = reward_scaling * batch["reward"] + gamma * value_

    def critic1_loss_fn(critic1_params):
        q1 = params.critic1.apply_fn(critic1_params, batch["state"], batch["action"])
        loss = 0.5 * jnp.mean((q1 - q_hat) ** 2)
        return loss

    grads1 = jax.grad(critic1_loss_fn)(params.critic1.params)
    new_critic1 = params.critic1.apply_gradients(grads=grads1)

    def critic2_loss_fn(critic2_params):
        q2 = params.critic2.apply_fn(critic2_params, batch["state"], batch["action"])
        loss = 0.5 * jnp.mean((q2 - q_hat) ** 2)
        return loss

    grads2 = jax.grad(critic2_loss_fn)(params.critic2.params)
    new_critic2 = params.critic2.apply_gradients(grads=grads2)

    return new_critic1, new_critic2
