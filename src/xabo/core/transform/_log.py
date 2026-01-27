import jax.numpy as jnp

from .._types import Scalar
from ._transform import Transform


class Log(Transform[Scalar]):
    """
    forward: x -> exp(x)
    inverse: y -> log(y)
    """

    def forward(self, unconstrained: Scalar) -> Scalar:
        return jnp.exp(unconstrained)

    def inverse(self, constrained: Scalar) -> Scalar:
        return jnp.log(constrained)

    def log_det_jacobian(self, unconstrained: Scalar) -> Scalar:
        # d/dx(exp(x)) = exp(x), so log|J| = log(exp(x)) = x
        return unconstrained
