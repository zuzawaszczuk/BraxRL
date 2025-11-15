from Actor import Actor
from Critic import Critic
from Memory import Memory
from typing import Tuple
import jax


class Agent:
    def __init__(self, 
                observation_dim: int,
                actions_number: int, 
                hidden_dim: int = 64,
                gamma: float = 0.99, 
                _lambda: float = 0.95, 
                policy_clip: float = 0.2, 
                batch_size: int = 64,
                epochs: int = 10
                 ):
        self.gamma = gamma
        self._lambda = _lambda
        self.policy_clip = policy_clip
        self.epochs = epochs

        self.actor = Actor(observation_dim = observation_dim, 
                           hidden_dim = hidden_dim, 
                           action_dim = actions_number)
        
        self.critic = Critic(observation_dim = observation_dim,
                             hidden_dim = hidden_dim)
        
        self.memory = Memory(batch_size = batch_size)

    def remember(self, state, prob, val, action, reward, done) -> None:
        self.memory.store_memory(state, prob, val, action, reward, done)

    def choose_action(self, observation) -> Tuple[jax.Array, jax.Array, jax.Array]:
        #   TODO
        pass

    def learn(self):
        #   TODO
        pass