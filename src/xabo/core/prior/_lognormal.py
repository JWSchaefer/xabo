from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from .._types import Scalar, T
from ._prior import Prior


@dataclass(frozen=True)
class LogNormal(Prior[T]):
    """Log-normal prior: log(X) ~ Normal(mu, sigma).

    Suitable for positive parameters like lengthscales and variances.
    """

    mu: float = 0.0
    sigma: float = 1.0

    def log_prob(self, value: T) -> Scalar:
        """Log probability density at value in constrained space.

        For arrays, returns the sum of element-wise log probs.
        """
        log_x = jnp.log(value)
        element_log_prob = (
            -log_x
            - jnp.log(self.sigma)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * ((log_x - self.mu) / self.sigma) ** 2
        )
        return jnp.sum(element_log_prob)

    def sample(self, rng_key, shape=()) -> Array:
        """Sample from prior (returns constrained value).

        Args:
            rng_key: JAX random key
            shape: Output shape (default: scalar)
        """
        z = jr.normal(rng_key, shape=shape)
        return jnp.exp(self.mu + self.sigma * z)
