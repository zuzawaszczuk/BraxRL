import jax
from flax import linen as nn
import numpy as np


class Actor(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        policy = nn.Dense(self.observation_dim)(state)
        policy = nn.tanh(policy)
        policy = nn.Dense(self.hidden_dim)(policy)
        policy = nn.tanh(policy)
        policy = nn.Dense(self.action_dim)(policy)
        return policy
