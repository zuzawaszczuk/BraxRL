import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float


class ValueNetwork(nn.Module):
    observation_size: int
    fc1_dims: int = 256
    fc2_dims: int = 256

    @nn.compact
    def __call__(self, state: Array) -> Array:
        assert state.shape[-1] == self.observation_size

        x = nn.Dense(self.fc1_dims)(state)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)
