import jax.numpy as jnp
from typing import Tuple, List


class Memory:
    def __init__(self, batch_size: int):
        self.states:      list = []
        self.probs:       list = []
        self.vals:        list = []
        self.actions:     list = []
        self.rewards:     list = []
        self.dones:       list = []
        self.batch_size = batch_size

    def generate_batches(self) -> Tuple[jnp.ndarray, 
                                        jnp.ndarray, 
                                        jnp.ndarray, 
                                        jnp.ndarray, 
                                        jnp.ndarray, 
                                        jnp.ndarray, 
                                        List[jnp.ndarray]]:
        states_number = len(self.states)
        batch_start = jnp.arange(0, states_number, self.batch_size)
        indices = jnp.arange(states_number, dtype=jnp.int64)
        jnp.random.shuffle(indices)
        batches = [indices[index : index + self.batch_size] for index in batch_start]

        return (jnp.array(self.states),
                jnp.array(self.probs),
                jnp.array(self.vals),
                jnp.array(self.actions),
                jnp.array(self.rewards),
                jnp.array(self.dones),
                batches)
    
    def store_memory(self, state, prob, val, action, reward, done) -> None:
        self.states.append(state), 
        self.probs.append(prob), 
        self.vals.append(val),
        self.actions.append(action), 
        self.rewards.append(reward), 
        self.dones.append(done)

    def clear_memory(self) -> None:
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
