from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from jaxtyping import Array, PRNGKeyArray


class ActorNetwork(nn.Module):
    observation_size: int
    action_size: int
    fc1_dims: int = 256
    fc2_dims: int = 256
    reparam_noise: float = 1e-6

    @nn.compact
    def __call__(self, state: Array) -> Tuple[Array, Array]:
        assert state.shape[-1] == self.observation_size

        prob = nn.Dense(self.fc1_dims)(state)
        prob = nn.relu(prob)
        prob = nn.Dense(self.fc2_dims)(prob)
        prob = nn.relu(prob)

        mu = nn.Dense(self.action_size)(prob)
        sigma = nn.Dense(self.action_size)(prob)

        sigma = jax.lax.clamp(self.reparam_noise, sigma, float(1))

        return mu, sigma


def sample_normal(
    actor: TrainState,
    state: Array,
    max_action: int,
    key: PRNGKeyArray,
    reparameterize: bool = True,
    reparam_noise: float = 1e-6,
) -> Tuple[Array, Array]:

    mu, sigma = actor.apply_fn(actor, state)
    dist = distrax.Normal(loc=mu, scale=sigma)

    if reparameterize:
        actions = dist.sample(seed=key)
    else:
        actions = jax.lax.stop_gradient(dist.sample(seed=key))

    action = jax.lax.tanh(actions) * max_action

    log_probs = dist.log_prob(actions)
    log_probs -= jnp.log(1 - jnp.square(action) + reparam_noise)
    log_probs = log_probs.sum(1, keepdim=True)

    return action, log_probs
