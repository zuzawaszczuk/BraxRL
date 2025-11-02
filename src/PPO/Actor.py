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




# critic = Actor(observation_dim = 120, hidden_dim = 64, action_dim=4)
# x = np.random.uniform(low = 0, high = 12, size = (4, 4))
# key = jax.random.key(0)
# params = critic.init(key, x)
# y = critic.apply(params, x)
# print(y)