from abc import ABC, abstractmethod
from typing import TypeVar

from beartype import beartype

from ..spec import Spec
from ._types import KernelInputA, KernelInputB, KernelOutput

# TypeVars for kernel params and state
P = TypeVar('P')  # Params type
S = TypeVar('S')  # State type


@beartype
class Kernel(Spec[P, S], ABC):
    """Base class for kernel functions.

    Kernels are Specs that define covariance functions for GPs.
    Generic over P (params type) and S (state type) for typed access.
    """

    @abstractmethod
    def __call__(
        self,
        state: S,
        params: P,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        """Evaluate kernel function.

        Args:
            state: Typed state values
            params: Typed parameter values
            x: First input array, shape [..., A, X]
            x_tick: Second input array, shape [..., B, X]

        Returns:
            Kernel matrix, shape [..., A, B]
        """
        ...
