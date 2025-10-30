import jax
import numpy as np
from jax import numpy as jnp
from dataclasses import replace
from flax import struct


@struct.dataclass
class ReplayBuffer:
    capacity:int
    state_shape:tuple[int]
    action_shape:tuple[int]
    ptr:int = 0
    size:int = 0
    buffer:dict=None

    @classmethod
    def create(cls, capacity, state_shape, action_shape):
        buffer = {
            "state": jnp.zeros((capacity, state_shape), dtype=jnp.float32),
            "action": jnp.zeros((capacity, action_shape), dtype=jnp.float32),
            "new_state": jnp.zeros((capacity, state_shape), dtype=jnp.float32),
            "reward": jnp.zeros((capacity,), dtype=np.float32),
            "done": jnp.zeros((capacity,), dtype=jnp.float32)
        }
        return cls(capacity, state_shape, action_shape, buffer=buffer)

    def add(self, batch):
        batch_size = batch["state"].shape[0]
        idxs = (jnp.arange(batch_size) + self.ptr) % self.capacity

        buffer = {
        "state": self.buffer["state"].at[idxs].set(batch["state"]),
        "action": self.buffer["action"].at[idxs].set(batch["action"]),
        "new_state": self.buffer["new_state"].at[idxs].set(batch["new_state"]),
        "reward": self.buffer["reward"].at[idxs].set(batch["reward"]),
        "done": self.buffer["done"].at[idxs].set(batch["done"]),
        }

        new_ptr = (self.ptr + batch_size) % self.capacity
        new_size= jnp.minimum(self.size+batch_size, self.capacity)
        return replace(self, capacity=self.capacity, state_shape=self.state_shape, action_shape=self.action_shape, buffer=buffer, ptr=new_ptr, size=new_size)

    def sample(self, batch_size, subkey):
        idx = jax.random.randint(subkey, (batch_size), 0, self.size)
        return{
            "state": self.buffer["state"][idx],
            "action": self.buffer["action"][idx],
            "new_state": self.buffer["new_state"][idx],
            "reward": self.buffer["reward"][idx],
            "done": self.buffer["done"][idx],
        }
