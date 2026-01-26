from typing import Generic, Type, TypeVar

from xabo.core.spec._default import Default

T = TypeVar('T')


class Transform(Generic[T]):
    pass


class Prior(Generic[T]):
    pass


class Parameter(Default[T]):
    transform: Transform[T]
    prior: Prior[T]
