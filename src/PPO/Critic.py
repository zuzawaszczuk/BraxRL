import jax
from flax import linen as nn


class Critic(nn.Module):
    observation_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        value = nn.Dense(self.observation_dim)(state)
        value = nn.tanh(value)
        value = nn.Dense(self.hidden_dim)(value)
        value = nn.tanh(value)
        value = nn.Dense(1)(value)
        return value
