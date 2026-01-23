from typing import Generic, List, Self, TypeVar

from ..constraints import Constraint

T = TypeVar('T')


class Parameter(Generic[T]):
    _raw: T
    constrains: List[Constraint[T]]

    @property
    def value(self: Self) -> T:
        ...

    @value.setter
    def value(self, value: T):
        ...

    @property
    def raw_value(self):
        ...

    @raw_value.setter
    def raw_value(self, value: T):
        self._raw = value
