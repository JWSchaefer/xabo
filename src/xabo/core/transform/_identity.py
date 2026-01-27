import jax.numpy as jnp

from .._types import Scalar
from ._transform import Transform
from ._types import T


class Identity(Transform[T]):
    """Identity transform for unconstrained parameters."""

    def forward(self, unconstrained: T) -> T:
        return unconstrained

    def inverse(self, constrained: T) -> T:
        return constrained

    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        return jnp.zeros(())
