import jax.numpy as np
from beartype import beartype

from xabo.core.typing import typecheck

from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput


@beartype
class Matern12(Kernel):

    rho: float
    sigma: float

    def __init__(self, lengthscale: float, variance: float) -> None:
        self.rho = lengthscale
        self.sigma = variance

    @typecheck
    def __call__(
        self: 'Matern12',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        d = np.sum(
            np.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (self.sigma**2) * np.exp(-d / self.rho)
