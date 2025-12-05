from flax import linen as nn
import jax.numpy as jnp
import distrax
from jaxtyping import PRNGKeyArray, Float, Array
import jax
from flax.core.frozen_dict import FrozenDict
from typing import Tuple, Callable

# batch_size can be variable, state_dim and action_dim come from environment
State = Float[Array, "batch state_dim"]
Action = Float[Array, "batch action_dim"]
Params = FrozenDict


class ActorNetwork(nn.Module):
    lr: float
    input_dims: int
    n_actions: int
    fc1_dims: int
    fc2_dims: int
    reparam_noise: float

    @nn.compact
    def __call__(self, state: State) -> Tuple[Array, Array]:
        prob = nn.Dense(self.fc1_dims)(state)
        prob = nn.relu(prob)
        prob = nn.Dense(self.fc2_dims)(prob)
        prob = nn.relu(prob)

        mu = nn.Dense(self.n_actions)(prob)
        sigma = nn.Dense(self.n_actions)(prob)

        sigma = jax.lax.clamp(self.reparam_noise, sigma, 1)

        return mu, sigma


def sample_normal(
    apply: Callable[[Params, State], tuple[Array, Array]],
    model_params: Params,
    state: State,
    max_action: int,
    reparam_noise: float,
    key: PRNGKeyArray,
    reparameterize: bool = True,
) -> Tuple[Action, Action]:

    mu, sigma = apply(model_params, state)
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
