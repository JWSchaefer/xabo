from abc import ABC, abstractmethod

from beartype import beartype

from ..spec import Spec
from ._types import KernelInputA, KernelInputB, KernelOutput


@beartype
class Kernel(Spec, ABC):
    """Base class for kernel functions.

    Kernels are Specs that define covariance functions for GPs.
    """

    @abstractmethod
    def __call__(
        self: 'Kernel',
        state: dict,
        params: dict,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        """Evaluate kernel function.

        Args:
            params: Parameter values
            state:  State values
            x: First input array, shape [..., A, X]
            x_tick: Second input array, shape [..., B, X]

        Returns:
            Kernel matrix, shape [..., A, B]
        """
        ...
