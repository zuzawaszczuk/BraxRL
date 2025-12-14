import jax
import jax.numpy as jnp
from matplotlib.pyplot import step
import optax
from flax.training.train_state import TrainState
from replay_buffer import ReplayBuffer
from brax.envs import create


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
        return jnp.mean((q - y) ** 2)

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


def train_ddpg(actor, critic, env_name, num_timesteps=1000, reward_scaling=1.0, batch_size=8,
               learning_rate=1e-3, gamma=0.99, tau=0.005, buffer_capacity=100000,
               exploration_noise=0.2):

    key = jax.random.PRNGKey(0)
    # Split key for env reset and model init
    key, env_key, actor_key, critic_key = jax.random.split(key, 4)
    env = create(env_name=env_name)
    state = env.reset(rng=env_key)
    obs = state.obs
    ep_reward = 0.0

    obs_dim = state.obs.shape[0]
    action_dim = env.action_size

    # Initialize params
    dummy_obs = jnp.ones((obs_dim,))
    dummy_action = jnp.ones((action_dim,))
    actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs, dummy_action)

    target_actor_params = actor_params
    target_critic_params = critic_params

    # Initialize training states
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate))
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate))

    # Initialize ReplayBuffer
    replay_buffer = ReplayBuffer.create(obs_dim=obs_dim, act_dim=action_dim, capacity=buffer_capacity)

    @jax.jit
    def select_action(params, obs, key):
        action = actor.apply(params, obs)
        noise = jax.random.normal(key, shape=action.shape) * exploration_noise
        return jnp.clip(action + noise, -1.0, 1.0)

    @jax.jit
    def train_step(actor_state, critic_state, target_actor_params, target_critic_params, replay_buffer, key):
        key, sample_key = jax.random.split(key)
        s, a, r, ns, d = replay_buffer.sample(sample_key, batch_size)

        critic_state = update_critic(critic_state, target_critic_params,
                                     target_actor_params, actor_state, s, a, r, ns, d, gamma)
        actor_state = update_actor(actor_state, critic_state, s)

        target_actor_params = soft_update(target_actor_params, actor_state.params, tau)
        target_critic_params = soft_update(target_critic_params, critic_state.params, tau)

        return actor_state, critic_state, target_actor_params, target_critic_params, key

    jit_env_step = jax.jit(env.step)

    print("Training started")
    step_rewards = []
    episode_rewards = []

    for step in range(num_timesteps):

        if state.done:
            episode_rewards.append(ep_reward)
            print(f'\n Steps: {step+1} Reward: {ep_reward:.2f}')
            key, env_key = jax.random.split(key)
            state = env.reset(rng=env_key)
            obs = state.obs
            ep_reward = 0.0

        key, action_key = jax.random.split(key)
        action = select_action(actor_state.params, obs, action_key)
        new_state = jit_env_step(state, action)
        obs = state.obs
        reward = new_state.reward * reward_scaling
        done = new_state.done
        next_obs = new_state.obs
        replay_buffer = replay_buffer.push(
            state=obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done
        )
        step_rewards.append(float(reward))

        state = new_state
        ep_reward += float(reward)
        if replay_buffer.size >= batch_size:
            actor_state, critic_state, target_actor_params, target_critic_params, key = train_step(
                actor_state, critic_state, target_actor_params, target_critic_params, replay_buffer, key
            )


    print("Training complete.")
    return actor_state.params, critic_state.params, step_rewards, episode_rewards
