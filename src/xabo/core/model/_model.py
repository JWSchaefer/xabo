from abc import ABC, abstractmethod
from typing import Any, Self


class Model(ABC):

    _trained: bool

    def __init__(self: Self):
        self._trained = False

    @abstractmethod
    def __call__(self: Self, x: Any) -> Any:
        ...

    @abstractmethod
    def train(cls: Self, x: Any, y: Any):
        ...

    @property
    def is_trained(self) -> bool:
        return self._trained

    def set_trained(self):
        self._trained = True

    def set_untrained(self):
        self._trained = False
