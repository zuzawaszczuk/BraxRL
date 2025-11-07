import optax
from flax import linen as nn
import jax.numpy as jnp
import orbax.checkpoint as ocp

class CriticNetwork(nn.Module):
    lr: float
    input_dims: int
    n_actions: int
    fc1_dims: int
    fc2_dims: int
    checkpoint_dir: str
    name: str

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.fc1_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
    
    def save_checkpoint(self, train_state, step: int):
        ckpt = ocp.CheckpointManager(self.checkpoint_dir)
        ckpt.save(step, train_state)

    def load_checkpoint(self, train_state):
        ckpt = ocp.CheckpointManager(self.checkpoint_dir)
        restored = ckpt.restore(ocp.latest_step(self.checkpoint_dir), train_state)
        return restored
