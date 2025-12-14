import timeit
import numpy as np
import jax.numpy as jnp
from jax import random, device_put


key = random.PRNGKey(0)
size = 3000

x_jnp = random.normal(key, (size, size), dtype=jnp.float32)
x_np = np.random.normal(size=(size, size)).astype(np.float32)
x_np_device = device_put(x_np)


def jax_dot_devicearray():
    return jnp.dot(x_jnp, x_jnp.T).block_until_ready()


def numpy_dot_cpu():
    return np.dot(x_np, x_np.T)


def jax_dot_numpy_transfer():
    return jnp.dot(x_np, x_np.T).block_until_ready()


def jax_dot_numpy_device():
    return jnp.dot(x_np_device, x_np_device.T).block_until_ready()


number = 10
repeat = 10

print("Timing JAX array on GPU (fast):")
print(timeit.repeat(jax_dot_devicearray, number=number, repeat=repeat))

print("\nTiming NumPy array on CPU (slow):")
print(timeit.repeat(numpy_dot_cpu, number=number, repeat=repeat))

print("\nTiming JAX dot with NumPy array (implicit transfer):")
print(timeit.repeat(jax_dot_numpy_transfer, number=number, repeat=repeat))

print("\nTiming JAX dot with NumPy array on device (explicit transfer):")
print(timeit.repeat(jax_dot_numpy_device, number=number, repeat=repeat))
