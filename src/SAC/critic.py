from flax import linen as nn
import jax.numpy as jnp

class CriticNetwork(nn.Module):
    # input_dims: int  [0] should have dim of state
    # n_actions: int
    fc1_dims: int
    fc2_dims: int

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray: # can set up data type mixed precision
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.fc1_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x