import jax
import numpy as np
from jax import numpy as jnp

class ReplayBuffer:
    """
    Class to hold observation from Mujocco environment.

    """
    def __init__(self, capacity, state_shape, action_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.action = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.newstate = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, new_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.newstate[self.ptr] = new_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, key):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size), 0, self.size)
        return (
            jnp.array(self.state[idx], dtype=jnp.float32),
            jnp.array(self.action[idx], dtype=jnp.float32),
            jnp.array(self.rewards[idx], dtype=jnp.float32),
            jnp.array(self.newstate[idx], dtype=jnp.float32),
            jnp.array(self.dones[idx], dtype=jnp.float32)
        )