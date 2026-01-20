from abc import ABC, abstractmethod
from typing import Self

from xabo.core.mean._types import MeanVector


class Mean(ABC):
    @abstractmethod
    def __call__(self: Self, x: MeanVector) -> MeanVector:
        pass
