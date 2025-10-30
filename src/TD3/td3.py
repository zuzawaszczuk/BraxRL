import jax
import optax
import copy
from mlp import MLP
from jax import jit
from jax import numpy as jnp
from flax.training.train_state import TrainState
from flax.core import unfreeze
from brax.envs import create
from functools import partial
from jax import lax
from replaybuffer import ReplayBuffer

def TD3_train(env_name:str, timesteps:int, evals:int, reward_scalling:int, action_before:int, normalize_obs:bool,
               action_repeat:int, discounting:float, lr_actor:float, lr_critic:float, num_envs:int, batch_size:int,
               num_grad_update:int, ro, min_replay_size:int, max_replay_size:int, seed:int, add_size:int=500, actor_size:list[int]=[256, 256],
                 critic_size:list[int]=[256, 256], actor_update:int=2, noise_clip:float=0.2, action_clip:int=1, checkpoint_path:str=None):

    @jit
    def add_actions(state):
        i, val, loopkey, uniform_flag, rep_buff = state
        i = i + num_envs * action_before
        loopkey, get_actions_key = jax.random.split(loopkey)
        rep_buff = get_actions(get_actions_key, envs, num_envs, actor_s, rep_buff, action_before,
                     action_space, reward_scalling, uniform_flag
        )

        return i, val, loopkey, uniform_flag, rep_buff

    @jit
    def while_condition(state):
        i, val, _, _, _ = state
        return i < val

    @jit
    def loop_update_critic(data):
        i, actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, critic_key, batch = data
        critic1_s, critic2_s = update_critics(critic_key, batch, critic1_s, critic2_s, discounting,
                                                    critic1_target, critic2_target, actor_s, actor_target,
                                                    noise_clip, action_clip)
        return i, actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, critic_key, batch

    @jit
    def true_condition(data):
        i, actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, critic_key, batch = data
        actor_s = update_actor(batch, actor_s, critic1_s)
        actor_target = polyak_averaging(actor_target, actor_s.params, ro)
        critic1_target = polyak_averaging(critic1_target, critic1_s.params, ro)
        critic2_target = polyak_averaging(critic2_target, critic2_s.params, ro)
        return i, actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, critic_key, batch

    @jit
    def update_data(i, data):
        actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, key, batch = data
        key, critic_key = jax.random.split(key)
        pred = lax.rem(i, actor_update)
        i, actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, _, _ = lax.cond(pred, true_condition, loop_update_critic, (i, actor_s, critic1_s, critic2_s, actor_target,
                                                                              critic1_target, critic2_target, critic_key, batch))

        return actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, key, batch


    @jit
    def timestep_loop(i, data):
        actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, loopkey, rep_buff = data
        loopkey, rep_key, innerloopkey, batch_key = jax.random.split(loopkey, 4)

        batch = rep_buff.sample(batch_size, batch_key)
        actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, _, _ = lax.fori_loop(0, num_grad_update,
                                                                                                       update_data, (actor_s, critic1_s, critic2_s, actor_target,
                                                                                                                     critic1_target, critic2_target,
                                                                                                                     innerloopkey, batch))
        add_state = (0, add_size, rep_key, False, rep_buff)

        _, _, _, _, rep_buff = lax.while_loop(while_condition, add_actions, add_state)

        return actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, loopkey, rep_buff



    env = create(env_name)
    state = env.reset(jax.random.PRNGKey(seed))
    obs_space = state.obs.shape[-1]
    action_space = env.action_size
    del env


    envs = create(env_name, action_repeat=action_repeat, batch_size=num_envs)

    _, actor_key, critic1_key, critic2_key, loopkey = jax.random.split(jax.random.PRNGKey(seed), 5)


    actor_s = create_trainstate(actor_key, obs_space, action_space, actor_size, optax.adam, lr_actor, True)
    critic1_s = create_trainstate(critic1_key, obs_space+action_space, 1, critic_size, optax.adam, lr_critic)
    critic2_s = create_trainstate(critic2_key, obs_space+action_space, 1, critic_size, optax.adam, lr_critic)

    actor_target = copy.deepcopy(unfreeze(actor_s.params))
    critic1_target = copy.deepcopy(unfreeze(critic1_s.params))
    critic2_target = copy.deepcopy(unfreeze(critic2_s.params))


    rep_buff = ReplayBuffer.create(max_replay_size, obs_space, action_space)

    initial_state = (0, min_replay_size, loopkey, True, rep_buff)

    _, _, _, _, rep_buff= lax.while_loop(while_condition, add_actions, initial_state)



    actor_s, critic1_s, critic2_s, actor_target, critic1_target, critic2_target, _, rep_buff= lax.fori_loop(0, timesteps, timestep_loop, (actor_s, critic1_s,
                                                                                                                                           critic2_s, actor_target,
                                                                                                                                           critic1_target, critic2_target,
                                                                                                                                           loopkey, rep_buff))
    return actor_s


@jit
def update_actor(batch, actor_s, critic1_s):

    def loss_actor(params, batch, critic_s):
        obs = batch["state"]
        action = actor_s.apply_fn(params, obs)
        q_value = critic_s.apply_fn(critic_s.params, jnp.concat([action, obs], axis=-1))

        return -jnp.mean(q_value)

    grad_fn = jax.grad(loss_actor)
    grads = grad_fn(actor_s.params, batch, critic1_s)
    new_actor = actor_s.apply_gradients(grads=grads)
    return new_actor


@jit
def update_critics(key, batch, critic1_s, critic2_s, gamma, critic1_target, critic2_target,
                   actor_s, actor_target, noise_clip, action_clip):

    action = actor_s.apply_fn(actor_target, batch["new_state"])
    action = corrupt_action(action, key, noise_clip, action_clip)

    x = jnp.concat([batch["new_state"], action], axis=-1)
    q1 = critic1_s.apply_fn(critic1_target, x)
    q2 = critic2_s.apply_fn(critic2_target, x)

    min_target = jnp.min(jnp.array([q1, q2]), axis=0)

    y = batch["reward"] + gamma * (1.0 - batch["done"]) * min_target

    def loss_critic1(params, batch, y):
        q_val = critic1_s.apply_fn(params, jnp.concat([batch["state"], batch["action"]], axis=-1))
        loss = jnp.mean(jnp.pow(q_val - y, 2))
        return loss

    def loss_critic2(params, batch, y):
        q_val = critic2_s.apply_fn(params, jnp.concat([batch["state"], batch["action"]], axis=-1))
        loss = jnp.mean(jnp.pow(q_val - y, 2))
        return loss

    grad1_fn = jax.grad(loss_critic1)
    grad2_fn = jax.grad(loss_critic2)

    grads1 = grad1_fn(critic1_s.params, batch, y)
    grads2 = grad2_fn(critic2_s.params, batch, y)

    new_critic1_s = critic1_s.apply_gradients(grads=grads1)
    new_critic2_s = critic2_s.apply_gradients(grads=grads2)

    return new_critic1_s, new_critic2_s


def create_trainstate(key, input_size:int,
                       output_size:int, hidden_layers:list[int],
                        optimizer, lr:float, actor=False) -> TrainState:
    mlp = MLP(hidden_layers, input_size, output_size, actor)
    params = mlp.init(key, jnp.zeros(input_size))
    return TrainState.create(apply_fn=mlp.apply, params=params, tx=optimizer(lr))


@jit
def corrupt_action(action,
                    subkey,
                    noise_clip:float,
                      action_clip:float):
    noise = jnp.clip(jax.random.normal(subkey, action.shape), -noise_clip, noise_clip)
    return jnp.clip(action + noise, -action_clip, action_clip)




# @partial(jit, static_argnums=(2, 5, 6, 8))
def get_actions(key, envs, env_num:int, actor_ts:TrainState, rep_buff,
                 action_before:int, action_space:int, reward_scalling:float=1,
                   uniform=False,
                ):


    def env_step(i, carry):
        key, state, rep_buff, actions = carry
        key, action_subkey = jax.random.split(key)
        actions = lax.cond(uniform, uniform_actions, actor_actions, (action_subkey, state.obs))
        new_state = envs.step(state, actions)


        obs, new_obs = state.obs, new_state.obs


        batch = {
            "state": obs,
            "action": actions,
            "new_state": new_obs,
            "reward": new_state.reward * reward_scalling,
            "done": new_state.done
        }


        rep_buff = rep_buff.add(batch)


        return key, new_state, rep_buff, actions

    def uniform_actions(carry):
        key, _ = carry
        return jax.random.uniform(key, (env_num, action_space), minval=-1, maxval=1)

    def actor_actions(carry):
        _, state = carry
        return actor_ts.apply_fn(actor_ts.params, state)


    key, env_subkey, loop_subkey, before_key = jax.random.split(key, 4)
    state = envs.reset(env_subkey)
    curr_action_before = jax.random.randint(before_key, (), 1, action_before+1)
    actions = jnp.zeros((env_num, action_space))
    loop_subkey, state, rep_buff, _= lax.fori_loop(0, curr_action_before, env_step, (loop_subkey, state, rep_buff, actions))
    return rep_buff

@jit
def polyak_averaging(params,
                 target_params,
                   ro:float):
    return jax.tree.map(lambda x, y: ro * x + (1 - ro) * y, target_params, params)
