from flax import linen as nn
import jax.numpy as jnp
from jaxtyping import Float, Array

# batch_size can be variable, state_dim and action_dim come from environment
State = Float[Array, "batch state_dim"]
Action = Float[Array, "batch action_dim"]
Value = Float[Array, "batch 1"]


class CriticNetwork(nn.Module):
    # input_dims: int  [0] should have dim of state
    # n_actions: int
    fc1_dims: int
    fc2_dims: int

    @nn.compact
    def __call__(self, state: State, action: Action) -> Value:
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.fc1_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
