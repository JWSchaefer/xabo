from abc import ABC, abstractmethod

from jaxtyping import PyTree

from ..spec import Spec
from ._types import KernelInputA, KernelInputB, KernelOutput


class Kernel(Spec, ABC):
    @abstractmethod
    def __call__(
        self: 'Kernel',
        state: PyTree,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        ...
