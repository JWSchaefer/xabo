from abc import ABC, abstractmethod
from typing import TypeVar

from beartype import beartype
from trellis import Spec

from ._types import MeanVector

P = TypeVar('P')
S = TypeVar('S')
Tr = TypeVar('Tr')


@beartype
class Mean(Spec[P, S, Tr], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: MeanVector,
    ) -> MeanVector:
        ...
