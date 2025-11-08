import jax
import jax.numpy as jnp
import flashbax as fbx
from critic import CriticNetwork


functools.partial(sac_train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, episode_length=1000, normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=512, grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1, save_checkpoint_path=f"reports/checkpoints/{env_name}"),


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

def save_checkpoint(self, train_state, step: int):
        ckpt = ocp.CheckpointManager(self.checkpoint_dir)
        ckpt.save(step, train_state)

    def load_checkpoint(self, train_state):
        ckpt = ocp.CheckpointManager(self.checkpoint_dir)
        restored = ckpt.restore(ocp.latest_step(self.checkpoint_dir), train_state)
        return restored

    def create_train_state(self, rng)
        params = self.init(rng, jnp.zeros((1, self.input_dims)), jnp.zeros((1, self.n_actions)))
        optimizer = optax.adam(self.lr)

        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)