from critic import CriticNetwork
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

critic = CriticNetwork(
    lr=0.002,
    input_dims=20,
    n_actions=5,
    fc1_dims=256,
    fc2_dims=256,
    checkpoint_dir="checkpoint",
    name="critic1",
)


print(critic)

rng = jax.random.PRNGKey(0)
params = critic.init(
    rng, jnp.zeros((1, critic.input_dims)), jnp.zeros((1, critic.n_actions))
)


sgd_opt = optax.sgd(0.002, 0.001)

state = train_state.TrainState.create(apply_fn=critic.apply, params=params, tx=sgd_opt)

print(jax.tree.map(lambda x: x.shape, params))
critic.save_checkpoint(state, 1)
