import jax
import jax.numpy as jnp
import flashbax as fbx
from critic import CriticNetwork


buffer = fbx.make_flat_buffer(
    max_length = int(1e6),      
    min_length = 2,   
    sample_batch_size = 2
)

example_timestep = {
    "state": jnp.zeros((10,)),
    "action": jnp.zeros((5,)),
    "reward": jnp.array(0.0),
    "next_state": jnp.zeros((10,)),
    "done": jnp.array(False),
}
example_second = {
    "state": jnp.zeros((10,)),
    "action": jnp.zeros((5,)),
    "reward": jnp.array(0.0),
    "next_state": jnp.zeros((10,)),
    "done": jnp.array(False),
}
buffer_state = buffer.init(example_timestep)

print(buffer_state)

buffer_state = buffer.add(buffer_state, example_second)

rng = jax.random.PRNGKey(42)
batch = buffer.sample(buffer_state, rng)

print(buffer_state)

critic = CriticNetwork(lr=0.002,
    input_dims=20,
    n_actions=5,
    fc1_dims= 256,
    fc2_dims= 256,
    checkpoint_dir = "checkpoint",
    name= "critic1")


print(critic)

rng = jax.random.PRNGKey(0)
params = critic.init(rng,
                     jnp.zeros((1, critic.input_dims)),
                     jnp.zeros((1, critic.n_actions)))

print(jax.tree.map(lambda x: x.shape, params))