from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr

from xabo.core._types import Scalar

from ._prior import Prior


@dataclass(frozen=True)
class LogNormal(Prior[Scalar]):
    """Log-normal prior: log(X) ~ Normal(mu, sigma).

    Suitable for positive parameters like lengthscales and variances.
    """

    mu: float = 0.0
    sigma: float = 1.0

    def log_prob(self, value: Scalar) -> Scalar:
        """Log probability density at value in constrained space."""
        log_x = jnp.log(value)
        return (
            -log_x
            - jnp.log(self.sigma)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * ((log_x - self.mu) / self.sigma) ** 2
        )

    def sample(self, rng_key) -> Scalar:
        """Sample from prior (returns constrained value)."""
        z = jr.normal(rng_key)
        return jnp.exp(self.mu + self.sigma * z)
