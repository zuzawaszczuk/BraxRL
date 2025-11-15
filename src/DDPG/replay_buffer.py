from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class ReplayBuffer:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_states: jnp.ndarray
    dones: jnp.ndarray
    size: int
    capacity: int
    pos: int

    @classmethod
    def create(cls, obs_dim, act_dim, capacity=100000):
        return cls(
            states=jnp.zeros((capacity, obs_dim)),
            actions=jnp.zeros((capacity, act_dim)),
            rewards=jnp.zeros((capacity,)),
            next_states=jnp.zeros((capacity, obs_dim)),
            dones=jnp.zeros((capacity,)),
            size=0,
            capacity=capacity,
            pos=0,
        )

    def push(self, state, action, reward, next_state, done):
        idx = self.pos % self.capacity
        new_buffer = self.replace(
            states=self.states.at[idx].set(state),
            actions=self.actions.at[idx].set(action),
            rewards=self.rewards.at[idx].set(reward),
            next_states=self.next_states.at[idx].set(next_state),
            dones=self.dones.at[idx].set(done),
            pos=(self.pos + 1) % self.capacity,
            size=jnp.minimum(self.size + 1, self.capacity)
        )
        return new_buffer

    def sample(self, key, batch_size):
        max_size = jnp.minimum(self.size, self.capacity)
        idx = jax.random.randint(key, (batch_size,), 0, max_size)
        batch = (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
        return batch
