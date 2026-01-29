from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple

from jax import Array

from .._types import Scalar, T


class Prior(ABC, Generic[T]):
    """Prior distribution over parameter values."""

    @abstractmethod
    def log_prob(self, value: T) -> Scalar:
        """Log probability density at value (in constrained space).

        For arrays, returns the sum of element-wise log probs.
        """
        ...

    @abstractmethod
    def sample(
        self,
        rng_key: Array,
        shape: Optional[Tuple[int, ...]],
    ) -> Array:
        """Sample from prior (returns constrained value).

        Args:
            rng_key: JAX random key
            shape: Output shape (default: scalar)
        """
        ...
