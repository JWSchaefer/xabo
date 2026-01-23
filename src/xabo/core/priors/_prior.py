from abc import ABC, abstractmethod

from beartype.typing import Self, TypeVar
from jax import Array

from ..parameters import Parameter

T = TypeVar('T')


class Prior(Parameter[T], ABC):

    _rng: Array

    @abstractmethod
    def sample(self: Self) -> None:
        ...

    @abstractmethod
    def log_prob(self: Self) -> float:
        ...
