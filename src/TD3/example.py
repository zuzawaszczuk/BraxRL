import td3 as t
from brax import envs
import jax
from brax.io import html
import datetime


now =datetime.datetime.now()
print(now)


actor_s = t.TD3_train("ant", 500, 4, 1, 6, True, 3, 0.99, 1e-4, 1e-3, 1, 256, 4, 0.98, 1000, 6000, 5, add_size=2000, actor_size=[256, 256], critic_size=[256, 256])
env = envs.create("ant", action_repeat=6, episode_length=4000)
states = []

now =datetime.datetime.now()
print(now)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(rng=jax.random.PRNGKey(0))
for i in range(4000):
    obs = state.obs
    state = jit_step(state, actor_s.apply_fn(actor_s.params, obs))
    states.append(state.pipeline_state)
    if state.done:
        print(i)
        break

html_vis = html.render(env.sys.tree_replace({'opt.timestep':env.dt}), states)
with open('example.html', 'w') as f:
    f.write(html_vis)

