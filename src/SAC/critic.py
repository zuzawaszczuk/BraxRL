import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class CriticNetwork(nn.Module):
    observation_size: int
    action_size: int
    fc1_dims: int = 256
    fc2_dims: int = 256

    @nn.compact
    def __call__(self, state: Array, action: Array) -> Array:
        assert state.shape[-1] == self.observation_size
        assert action.shape[-1] == self.action_size

        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.fc1_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)
