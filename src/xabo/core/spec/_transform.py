from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp

from xabo.core._types import Scalar

T = TypeVar('T')


class Transform(ABC, Generic[T]):
    """Bijective mapping between constrained and unconstrained spaces.

    - forward: unconstrained (R) -> constrained (e.g., R+)
    - inverse: constrained -> unconstrained
    - log_det_jacobian: for MCMC correction when sampling in unconstrained space
    """

    @abstractmethod
    def forward(self, unconstrained: T) -> T:
        """Map from unconstrained to constrained space."""
        ...

    @abstractmethod
    def inverse(self, constrained: T) -> T:
        """Map from constrained to unconstrained space."""
        ...

    @abstractmethod
    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        """Log determinant of Jacobian |d(forward)/d(unconstrained)|."""
        ...


class Identity(Transform[T]):
    """Identity transform for unconstrained parameters."""

    def forward(self, unconstrained: T) -> T:
        return unconstrained

    def inverse(self, constrained: T) -> T:
        return constrained

    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        return jnp.zeros(())


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
