import jax
import jax.numpy as jnp
import optax
from replay_buffer import ReplayBuffer
from flax.training.train_state import TrainState
import numpy as np

@jax.jit
def soft_update(target_params, source_params, tau):
    return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target_params, source_params)


@jax.jit
def update_critic(critic_state, target_critic_params, target_actor_params,
                  actor_state, batch_s, batch_a, batch_r, batch_ns, batch_d, gamma):
    def loss_fn(params):
        next_a = actor_state.apply_fn(target_actor_params, batch_ns)
        target_q = critic_state.apply_fn(target_critic_params, batch_ns, next_a)
        y = batch_r + gamma * (1. - batch_d) * jnp.squeeze(target_q)
        q = jnp.squeeze(critic_state.apply_fn(params, batch_s, batch_a))
        loss = jnp.mean((q - y) ** 2)
        return loss

    grads = jax.grad(loss_fn)(critic_state.params)
    return critic_state.apply_gradients(grads=grads)


@jax.jit
def update_actor(actor_state, critic_state, batch_s):
    def loss_fn(params):
        a = actor_state.apply_fn(params, batch_s)
        q = critic_state.apply_fn(critic_state.params, batch_s, a)
        return -jnp.mean(q)

    grads = jax.grad(loss_fn)(actor_state.params)
    return actor_state.apply_gradients(grads=grads)


def train_ddpg(actor, critic, env, episodes=100, max_steps=1000, batch_size=64,
               learning_rate=1e-3, gamma=0.99, tau=0.005, buffer_capacity=1000,
               exploration_noise=0.1):

    key = jax.random.PRNGKey(0)
    # Split key for env reset and model init
    key, env_key, actor_key, critic_key = jax.random.split(key, 4)
    state = env.reset(rng=env_key)

    obs_dim = state.obs.shape[0]
    action_dim = env.action_size

    # Init params
    dummy_obs = jnp.ones((obs_dim,))
    dummy_action = jnp.ones((action_dim,))
    actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs, dummy_action)

    target_actor_params = actor_params
    target_critic_params = critic_params

    # Init states
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate))
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate))

    @jax.jit
    def select_action_jit(params, obs, key):
        action = actor_state.apply_fn(params, obs)
        noise = jax.random.normal(key, shape=(action_dim,)) * exploration_noise
        action = jnp.clip(action + noise, -1.0, 1.0)
        return action

    replay_buffer = ReplayBuffer(buffer_capacity)
    episode_rewards = []
    print("Training started")

    for ep in range(episodes):
        # Split key for episode reset
        key, env_key = jax.random.split(key)
        state = env.reset(rng=env_key)

        obs = state.obs
        ep_reward = 0

        for step in range(max_steps):
            print(f'\rEpisode {ep+1}, Step {step+1}', end='')

            key, action_key = jax.random.split(key)
            action = select_action_jit(actor_state.params, obs, action_key)

            state = env.step(state, action)
            next_obs = state.obs
            reward = state.reward
            done = state.done

            replay_buffer.push(np.array(obs), np.array(action), float(reward),
                               np.array(next_obs), float(done))

            obs = next_obs
            ep_reward += reward

            if len(replay_buffer) >= batch_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)

                critic_state = update_critic(critic_state, target_critic_params,
                                             target_actor_params, actor_state, s, a, r, ns, d, gamma)
                actor_state = update_actor(actor_state, critic_state, s)

                target_actor_params = soft_update(target_actor_params, actor_state.params, tau)
                target_critic_params = soft_update(target_critic_params, critic_state.params, tau)

            if done:
                break

        episode_rewards.append(ep_reward)
        print(f'\nEpisode {ep+1}, Reward: {ep_reward:.2f}, Steps: {step+1}')

    return actor_state.params, critic_state.params, episode_rewards