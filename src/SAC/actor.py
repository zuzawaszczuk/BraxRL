from flax import linen as nn
import jax.numpy as jnp
import distrax

PRNGKey = jax.Array


class ActorNetwork(nn.Module):
    lr: float
    input_dims: int
    n_actions: int
    fc1_dims: int
    fc2_dims: int
    reparam_noise: float
    max_action: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        prob = nn.Dense(self.fc1_dims)(state)
        prob = nn.relu(prob)
        prob = nn.Dense(self.fc2_dims)(prob)
        prob = nn.relu(prob)

        mu = nn.Dense(self.n_action)(prob)
        sigma = nn.Dense(self.n_action)(prob)

        sigma = jax.lax.clamp(min=slef.reparam_noise, sigma, max=1)

        return mu, sigma


def sample_normal(model: ActorNetwork, state: jnp.ndarray, key: PRNGKey, reparameterize: bool=True) -> jnp.ndarray:
    mu, sigma = model.apply(state)
    dist = distrax.Normal(loc=mu, scale=sigma)
    
    if reparameterize:
        actions = dist.sample(seed=key)
    else:
        actions = jax.lax.stop_gradient(dist.sample(seed=key))

    action = jax.lax.tanh(actions) * self.max_action

    # pytorch code
    log_probs = probabilities.log_prob(actions)
    log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)

    log_probs = log_probs.sum(1, keepdim=True)

    return action, log_probs