from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from xabo.core._types import Scalar

T = TypeVar('T')


class Prior(ABC, Generic[T]):
    """Prior distribution over parameter values."""

    @abstractmethod
    def log_prob(self, value: T) -> Scalar:
        """Log probability density at value (in constrained space)."""
        ...

    @abstractmethod
    def sample(self, rng_key) -> T:
        """Sample from prior (returns constrained value)."""
        ...
