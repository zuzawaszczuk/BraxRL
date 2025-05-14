import jax
import time
from models.TD3.replay_buffer import ReplayBuffer
from models.TD3.mlp import MLP
from jax import jit
from flax.training.train_state import TrainState
from jax import numpy as jnp
from brax.envs import create


def add_action(replaybuffer: ReplayBuffer, env_name, num_env, actor_trainstate, size,
               action_before, action_std=0.1, key=jax.random.PRNGKey(int(time.time())), mean=None, var=None, obs_count=None):

    @jit
    def normalize(obs, mean, var):
        return (obs - mean)/jnp.sqrt(var+1e-8)

    @jit
    def update_stats(mean, var, count, new_obs):
        batch_mean = jnp.mean(new_obs, axis=0)
        batch_var = jnp.var(new_obs, axis=0)

        new_count = count + num_env
        mean = (mean * count + batch_mean) / new_count
        var = (var * count + batch_var) / new_count

        return mean, var, new_count

    env = create(env_name, batch_size=num_env)
    state = env.reset(key)
    obs_space = state.obs.shape[-1]
    action_space = env.action_size


    count = obs_count if obs_count is not None else 0
    local_mean = mean if mean is not None else jnp.zeros(obs_space)
    local_var = var if var is not None else jnp.ones(obs_space)

    counter = 0
    while counter < size:
        key, subkey = jax.random.split(key)
        state = env.reset(subkey)
        for _ in range(action_before):
            subkey, key = jax.random.split(key)
            actions = jax.random.uniform(subkey, (num_env, action_space), minval=-1, maxval=1)
            state = env.step(state, actions)
            local_mean, local_var, count = update_stats(local_mean, local_var, count, state.obs)

        key, subkey = jax.random.split(key)
        normal_noise = jax.random.normal(subkey, (num_env, action_space)) * action_std
        normalized_obs = normalize(state.obs, local_mean, local_var)
        actions = actor_trainstate.apply_fn(actor_trainstate.params, normalized_obs)
        actions = jnp.clip(actions+normal_noise, -1, 1)
        new_state = env.step(state, actions)
        normalized_new_obs = normalize(new_state.obs, local_mean, local_var)
        for i in range(num_env):
            replaybuffer.add(normalized_obs[i], actions[i], normalized_new_obs[i], new_state.reward[i], new_state.done[i])
        counter += num_env
    return replaybuffer, local_mean, local_var, count


def soft_update(params, target_params, ro):
    return jax.tree_map(lambda x, y: ro * x + (1 - ro) * y, target_params, params)


def create_trainstate(key, hidden_layers, optimizer, lr,
                      input_size, output_size, actor=False):
    mlp = MLP(hidden_layers, input_size, output_size, actor)
    params = mlp.init(key, jnp.zeros((1, input_size)))
    tx = optimizer(lr)
    return TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)


def train_model(epochs, replay_buffer, batch_size,
                actor_s, critic_s1, critic_s2, actor_target,
                critic1_target, critic2_target, key, actor_update=2, loss_clip=0.01, action_clip=1, std=0.4, gamma=0.8, ro=0.9):
    key, batch_subkey = jax.random.split(key)
    key, critic_subkey = jax.random.split(key)
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size, batch_subkey)
        if epoch % actor_update != 0:
            critic_result = update_critic(critic_subkey, batch, actor_s, critic_s1, critic_s2,
                              actor_target, critic1_target, critic2_target, loss_clip, action_clip, gamma, std)
            critic_s1, critic_s2 = critic_result[0], critic_result[1]
        else:
            critic_result = update_critic(critic_subkey, batch, actor_s, critic_s1, critic_s2,
                              actor_target, critic1_target, critic2_target, loss_clip, action_clip, gamma, std)
            actor_result = update_actor(batch, critic_s1, critic_s2, actor_s, critic1_target, critic2_target, actor_target, ro)
            critic_s1, critic_s2 = critic_result[0], critic_result[1]
            actor_s, critic1_target, critic2_target, actor_target = actor_result[0], actor_result[1], actor_result[2], actor_result[3]
    return critic_s1, critic_s2, actor_s, critic1_target, critic2_target, actor_target


def action_corrupt(mi, key, std, loss_clip, action_clip):
    batch_size, action_dim = mi.shape
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (batch_size, action_dim)) * std
    return jnp.clip(mi + jnp.clip(eps, -loss_clip, loss_clip), -action_clip, action_clip)


@jit
def update_model(trainstate, grads):
    return trainstate.apply_gradients(grads=grads)


def update_actor(batch, critic_state1, critic_state2, actor_state,
                 critic_target1, critic_target2, actor_target, ro):

    def loss_actor(params):
        mi = actor_state.apply_fn(params, batch[0])
        q_val = critic_state1.apply_fn(critic_state1.params, jnp.concat([batch[0], mi], axis=1))
        return -jnp.mean(q_val)

    grad_fn_act = jax.value_and_grad(loss_actor)
    loss_act, grad = grad_fn_act(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grad)
    actor_target = soft_update(actor_state.params, actor_target, ro)
    critic_target1 = soft_update(critic_state1.params, critic_target1, ro)
    critic_target2 = soft_update(critic_state2.params, critic_target2, ro)
    return new_actor_state, critic_target1, critic_target2, actor_target


@jit
def update_critic(key, batch, actor_state, critic_state1, critic_state2,
                  actor_target, critic_target1, critic_target2, loss_clip, action_clip, gamma, std):

    def loss_critic(params, x, y):
        y_pred = critic_state1.apply_fn(params, x)
        return jnp.mean(jnp.pow((y_pred - y), 2))

    mi = actor_state.apply_fn(actor_target, batch[3])
    a = action_corrupt(mi, key, std, loss_clip, action_clip)
    x = jnp.concat([batch[3], a], axis=-1)
    target = jnp.min(jnp.stack([critic_state1.apply_fn(critic_target1, x),
                     critic_state2.apply_fn(critic_target2, x)], axis=0), axis=0)
    y = batch[2][:, None] + gamma * (1 - batch[4][:, None]) * target
    x = jnp.concat([batch[0], batch[1]], axis=-1)

    grad_fn = jax.value_and_grad(loss_critic)
    loss_critic1, grads1 = grad_fn(critic_state1.params, x, y)
    loss_critic2, grads2 = grad_fn(critic_state2.params, x, y)

    new_critic_s1 = critic_state1.apply_gradients(grads=grads1)
    new_critic_s2 = critic_state2.apply_gradients(grads=grads2)

    return new_critic_s1, new_critic_s2
