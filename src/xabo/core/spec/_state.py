from typing import Generic, Type, TypeVar

from xabo.core.spec._default import Default

T = TypeVar('T', bound=Default)


class State(Generic[T]):
    @classmethod
    def default(cls: Type['State[T]'], rng=None) -> T:
        return cls.default(rng=rng)
