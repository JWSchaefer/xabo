from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')


class Transform(Generic[T]):
    pass


class Prior(Generic[T]):
    pass


class Default(ABC, Generic[T]):
    @classmethod
    @abstractmethod
    def default(cls) -> T:
        raise NotImplementedError('Default method not implemented')


class Parameter(Default[T]):
    transform: Transform[T]
    prior: Prior[T]
