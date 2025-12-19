from abc import ABC, abstractmethod

from ._types import KernelInputA, KernelInputB, KernelOutput


class Kernel(ABC):
    @abstractmethod
    def __call__(
        self: 'Kernel',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        ...
