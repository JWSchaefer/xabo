from dataclasses import dataclass
from typing import Generic, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax._src.core import typecheck
from jax.random import PRNGKey
from jaxtyping import Array

from .._types import S, Scalar, T
from ._prior import Prior


@dataclass(frozen=True)
class LogNormal(Prior[T], Generic[T, S]):
    """Log-normal prior: log(X) ~ Normal(mu, sigma).

    Generic over T (mu/output type) and S (sigma type) to support:
    - LogNormal[Float[Array, 'D'], float] - vector means, shared sigma
    - LogNormal[float, Float[Array, 'D']] - scalar mean, vector sigma
    """

    mu: T
    sigma: S

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
        return jnp.sum(
            element_log_prob,
            axis=None
            if jnp.asarray(value).shape[-1] == jnp.asarray(self.mu)[0]
            else -1,
        )

    def sample(
        self,
        rng_key: Array,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """
        Sample from prior (returns constrained value).
        """
        z = jr.normal(
            rng_key,
            shape=(shape if shape is not None else tuple())
            + jnp.asarray(self.mu).shape,
        )
        return jnp.exp(self.mu + self.sigma * z)
