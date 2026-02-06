from abc import ABC, abstractmethod

from beartype import beartype
from trellis import Spec

from ._types import KernelInputA, KernelInputB, KernelOutput, P, S, Tr


@beartype
class Kernel(Spec[P, S, Tr], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        ...
