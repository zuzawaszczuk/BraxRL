from typing import List
import jax.numpy as jnp
import jax
from brax.envs import create


#   Compute advantages using Generalized Advantage Estimation.
@jax.jit
def compute_advantages(
    next_value: jax.Array,
    next_terminated: jax.Array,
    rewards: jax.Array,
    values: jax.Array,
    terminated: jax.Array,
    gamma: float,
    lambda_: float) -> jax.Array:

    T = values.shape[0]
    terminated = terminated.float()
    next_terminated = next_terminated.float()

    next_values = jnp.concatenate([values[1:], next_value[None, :]])
    next_terminated = jnp.concatenate([terminated[1:], next_terminated[None, :]])

    deltas = rewards + gamma * next_values * (1.0 - next_terminated) - values
    deltas_n = len(deltas)

    advantages = jnp.zeros(shape=(1, deltas_n))
    advantages[-1] = deltas[-1]

    for s in reversed(range(T - 1)):
        advantages[s] = deltas[s] + gamma * lambda_ * (1.0 - terminated[s + 1]) * advantages[s + 1]
    return advantages


@jax.jit
def get_minibatch_indices(batch_size: int, minibatch_size: int) -> List[jnp.ndarray]:
    num_minbatches = batch_size // minibatch_size
    indices = jax.random.permutation(batch_size).reshape(num_minbatches, batch_size)    
    return list(indices)


@jax.jit
def clip_objective_function(
        policy: jax.Array, 
        old_policy: jax.Array, 
        advantage: jax.Array, 
        epsilon: float = 0.2,
        zero_div_blocker: float = 1e-8) -> float:

    advantage = (advantage - advantage.mean()) / (advantage.std() + zero_div_blocker)

    ratio = policy / old_policy
    return jnp.mean(jnp.min(ratio * advantage, jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage))


@jax.jit
def critic_loss_function(critic_output: float, target_value: float, advantage_target: float) -> float:
    return jnp.square(critic_output - (target_value + advantage_target))


@jax.jit
def entropy_bonus(policy: jax.Array, epsilon: float = 1e-8, axis = -1) -> float:
    policy = jnp.clip(policy, epsilon, 1.0)
    return -jnp.sum(policy * jnp.log(policy), axis = axis)

