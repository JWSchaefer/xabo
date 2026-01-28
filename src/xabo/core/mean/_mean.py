from abc import ABC, abstractmethod

from beartype import beartype

from ..spec import Spec
from ._types import MeanVector, P, S


@beartype
class Mean(Spec[P, S], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: MeanVector,
    ) -> MeanVector:

        ...
