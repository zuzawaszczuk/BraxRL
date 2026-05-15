import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.device = device
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, s, a, r, ns, d):
        num_samples = s.shape[0]
        idx = torch.arange(self.ptr, self.ptr + num_samples) % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r.reshape(-1, 1)
        self.next_states[idx] = ns
        self.dones[idx] = d.reshape(-1, 1)
        self.ptr = (self.ptr + num_samples) % self.capacity
        self.size = min(self.size + num_samples, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )