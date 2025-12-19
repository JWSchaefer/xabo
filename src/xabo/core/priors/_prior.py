from abc import abstractmethod
from typing import Generic, List, Self, TypeVar

T = TypeVar('T')


class Constraint:
    pass


class Prior(Generic[T]):

    constrains: List[Constraint]

    @abstractmethod
    def sample(self: Self) -> T:
        ...

    @abstractmethod
    def log_prob(self: Self) -> float:
        ...

    @property
    def value(self: Self) -> T:
        ...
