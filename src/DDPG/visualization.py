import jax
import numpy as np
from brax.io import html


def visualize_policy(actor, params, env, episodes=1, max_steps=100):

    for ep in range(episodes):
        state = env.reset(rng=jax.random.PRNGKey(ep))
        rollout = []

        for _ in range(max_steps):
            obs = np.array(state.obs)
            action = np.array(actor.apply(params, obs))
            rollout.append(state)
            state = env.step(state, action)
            if state.done:
                break

        html_vis = html.render(env.sys, [s.pipeline_state for s in rollout])
        with open(f"episode_{ep+1}.html", "w") as f:
            f.write(html_vis)
        print(f"Saved episode_{ep+1}.html")