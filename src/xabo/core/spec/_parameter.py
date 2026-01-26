from typing import Generic, TypeVar

T = TypeVar('T')


class Transform(Generic[T]):
    pass


class Prior(Generic[T]):
    pass


class Parameter(Generic[T]):
    transform: Transform[T]
    prior: Prior[T]
