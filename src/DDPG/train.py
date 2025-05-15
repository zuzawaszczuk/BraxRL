import jax
import jax.numpy as jnp
import optax
from replay_buffer import ReplayBuffer
from flax.training.train_state import TrainState
import numpy as np


def soft_update(target_params, source_params, tau):
    return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target_params, source_params)


def update_critic(critic_state, critic, target_critic_params, target_actor_params,
                  actor, batch_s, batch_a, batch_r, batch_ns, batch_d, gamma):
    def loss_fn(params):
        next_a = actor.apply(target_actor_params, batch_ns)
        target_q = critic.apply(target_critic_params, batch_ns, next_a)
        y = batch_r + gamma * (1. - batch_d) * jnp.squeeze(target_q)
        q = jnp.squeeze(critic.apply(params, batch_s, batch_a))
        loss = jnp.mean((q - y) ** 2)
        return loss

    grads = jax.grad(loss_fn)(critic_state.params)
    return critic_state.apply_gradients(grads=grads)


def update_actor(actor_state, actor, critic, critic_params, batch_s):
    def loss_fn(params):
        a = actor.apply(params, batch_s)
        q = critic.apply(critic_params, batch_s, a)
        return -jnp.mean(q)

    grads = jax.grad(loss_fn)(actor_state.params)
    return actor_state.apply_gradients(grads=grads)


def train_ddpg(actor, critic, env, episodes=100, max_steps=1000, batch_size=64,
               learning_rate=1e-3, gamma=0.99, tau=0.005):
    key = jax.random.PRNGKey(0)
    state = env.reset(rng=key)

    obs_dim = state.obs.shape[0]
    action_dim = env.action_size

    # Init params
    dummy_obs = jnp.ones((obs_dim,))
    dummy_action = jnp.ones((action_dim,))
    actor_params = actor.init(key, dummy_obs)
    critic_params = critic.init(key, dummy_obs, dummy_action)

    target_actor_params = actor_params
    target_critic_params = critic_params

    # Init states
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate))
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate))

    replay_buffer = ReplayBuffer()
    episode_rewards = []
    print("Training started")
    for ep in range(episodes):
        rng, key = jax.random.split(key)
        state = env.reset(rng=key)
        obs = np.array(state.obs)
        ep_reward = 0

        for step in range(max_steps):
            action = np.array(actor.apply(actor_state.params, obs)) + 0.1 * np.random.randn(action_dim)
            action = np.clip(action, -1.0, 1.0)
            state = env.step(state, action)
            next_obs = np.array(state.obs)
            reward = state.reward
            done = state.done

            replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward

            if len(replay_buffer) >= batch_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                s = jnp.array(s)
                a = jnp.array(a)
                r = jnp.array(r)
                ns = jnp.array(ns)
                d = jnp.array(d)

                critic_state = update_critic(critic_state, critic, target_critic_params,
                                             target_actor_params, actor, s, a, r, ns, d, gamma)
                actor_state = update_actor(actor_state, actor, critic, critic_state.params, s)

                target_actor_params = soft_update(target_actor_params, actor_state.params, tau)
                target_critic_params = soft_update(target_critic_params, critic_state.params, tau)

            if done:
                print(f'Episode {ep+1} took: {step + 1:.2f} steps')
                break

        episode_rewards.append(ep_reward)
        print(f'Episode {ep+1}, Reward: {ep_reward:.2f}')

    return actor_state.params, critic_state.params, episode_rewards