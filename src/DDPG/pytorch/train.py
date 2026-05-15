from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
import torch
from nn import Actor, Critic
import torch.optim as optim
from replay_buffer import ReplayBuffer



class DDPGAgent:
    def __init__(self, obs_dim, action_dim, device, lr=1e-3, tau=0.005, gamma=0.99):
        self.device = device
        self.tau = tau
        self.gamma = gamma

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, buffer, batch_size):
        s, a, r, ns, d = buffer.sample(batch_size)
        with torch.no_grad():
            next_a = self.actor_target(ns)
            target_q = self.critic_target(ns, next_a)
            y = r + self.gamma * (1 - d) * target_q

        q_value = self.critic(s, a)
        critic_loss = F.mse_loss(q_value, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def sample_action(self, obs, noise_scale=0.2):
        with torch.no_grad():
            action = self.actor(obs)
            noise = torch.randn_like(action) * noise_scale
            return torch.clamp(action + noise, -1, 1)


def train(env_name="ant", num_timesteps=1_000_000, batch_size=256, device='cuda',
               reward_scaling=1.0, learning_rate=1e-3, gamma=0.99, tau=0.005, buffer_capacity=100000,
               exploration_noise=0.2, start_steps=10000, progress_callback=None, eval_freq=10000):
    env = envs.create(env_name=env_name, backend='spring')
    env = gym_wrapper.VectorGymWrapper(env)
    env = torch_wrapper.TorchWrapper(env, device=device)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_envs = env.num_envs

    agent = DDPGAgent(obs_dim, action_dim, device, lr=learning_rate, tau=tau, gamma=gamma)
    buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, device)

    obs = env.reset()

    episode_rewards = []
    episode_reward = 0

    for t in range(0, num_timesteps, num_envs):
        if t < start_steps:
            action = torch.rand((num_envs, action_dim), device=device) * 2 - 1
        else:
            action = agent.sample_action(obs, noise_scale=exploration_noise)
        next_obs, reward, done, info = env.step(action)
        buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs

        if buffer.size > batch_size:
            agent.update(buffer, batch_size)

        if done.any():
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs = env.reset()
            if progress_callback is not None:
                progress_callback(t, episode_rewards)
        else:
            episode_reward += reward.mean().item()

    return agent
