from abc import ABC, abstractmethod
from typing import TypeVar

from beartype import beartype

from ..spec import Spec
from ._types import KernelInputA, KernelInputB, KernelOutput

P = TypeVar('P')
S = TypeVar('S')


@beartype
class Kernel(Spec[P, S], ABC):
    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        ...
