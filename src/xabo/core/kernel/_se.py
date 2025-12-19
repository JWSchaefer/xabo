import jax.numpy as np
from beartype import beartype

from xabo.core.typing import typecheck

from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput


@beartype
class SquaredExponential(Kernel):

    ell: float
    sigma: float

    def __init__(self, lengthscale: float, variance: float) -> None:
        self.ell = lengthscale
        self.sigma = variance

    @typecheck
    def __call__(
        self: 'SquaredExponential',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        if len(x.shape) == 1:
            x = x[:, None]

        if len(x_tick.shape) == 1:
            x_tick = x_tick[:, None]

        diff = x[..., :, None, :] - x_tick[..., None, :, :]

        sqdist = np.sum(np.pow(diff, 2), axis=-1)

        out = self.sigma * np.exp(-0.5 * sqdist / (self.ell**2))

        return out
