from abc import ABC
from typing import Generic, TypeVar

T = TypeVar('T')


class Default(ABC, Generic[T]):
    @classmethod
    def default(cls, rng=None) -> T:
        raise NotImplementedError(
            f'Please implement the method `default` for your class `{cls.__name__}`'
        )
