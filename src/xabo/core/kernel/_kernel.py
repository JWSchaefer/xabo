from abc import ABC, abstractmethod

from beartype import beartype

from ..spec._spec import P, S, Spec, Tr
from ._types import KernelInputA, KernelInputB, KernelOutput


@beartype
class Kernel(Spec[P, S, Tr], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput: ...
