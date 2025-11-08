from flax import linen as nn
import jax.numpy as jnp

class ValueNetwork(nn.Module):
    fc1_dims: int
    fc2_dims: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.fc1_dims)(state)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x