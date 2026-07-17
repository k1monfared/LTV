"""Small deep MLP for LTV regression, implemented in JAX via flax.linen."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class LTVNet(nn.Module):
    """Feed-forward network predicting standardized log1p(LTV)."""

    hidden_sizes: tuple[int, ...] = (64, 32)

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)
