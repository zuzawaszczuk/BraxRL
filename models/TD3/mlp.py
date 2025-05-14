from flax import linen as nn


class MLP(nn.Module):
    """
    Multi Layer Perceptron
    """
    layers_sizes: list
    input_size: int
    output_size: int
    actor: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.input_size)(x)
        for size in self.layers_sizes:
            x = nn.Dense(size)(x)
            x = nn.gelu(x)
        if self.actor:
            x = nn.Dense(self.output_size, kernel_init=nn.initializers.uniform(scale=1e-3))(x)
            x = nn.tanh(x)
        else:
            x = nn.Dense(self.output_size, kernel_init=nn.initializers.uniform(scale=3e-3))(x)

        return x
