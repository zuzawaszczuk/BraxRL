env = envs.create(env_name=env_name, backend=backend)
inference_fn = make_inference_fn(params)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

html_string = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
with open(f"reports/visualizations/{env_name}.html", "w", encoding="utf-8") as f:
    f.write(html_string)
