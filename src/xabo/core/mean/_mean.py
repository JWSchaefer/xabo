from abc import ABC, abstractmethod

from beartype import beartype

from ..spec._spec import P, S, Spec, Tr
from ._types import MeanVector


@beartype
class Mean(Spec[P, S, Tr], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: MeanVector,
    ) -> MeanVector: ...
