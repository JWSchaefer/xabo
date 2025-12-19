from abc import ABC, abstractmethod
from typing import Any, Self, Type


class Model(ABC):
    @abstractmethod
    def __call__(self: Self, x: Any) -> Any:
        ...

    @classmethod
    @abstractmethod
    def train(cls: Type[Self], x: Any, y: Any) -> Self:
        ...
