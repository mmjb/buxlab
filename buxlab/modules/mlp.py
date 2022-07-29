from typing import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, dim in enumerate(self.dims):
            x = nn.Dense(dim, use_bias=self.use_bias)(x)
            if i != len(self.dims) - 1:
                x = self.activation(x)
        return x
