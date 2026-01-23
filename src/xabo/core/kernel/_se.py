import jax.numpy as np
from beartype import beartype
from jaxtyping import Float

from ..typing import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput


@beartype
class SquaredExponential(Kernel):

    ell: float | Float[np.ndarray, '...']
    sigma: float | Float[np.ndarray, '...']

    def __init__(
        self: 'SquaredExponential',
        lengthscale: float | Float[np.ndarray, '...'],
        variance: float | Float[np.ndarray, '...'],
    ):
        self.ell = lengthscale
        self.sigma = variance

    @typecheck
    def __call__(
        self: 'SquaredExponential',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        diff = x[..., :, None, :] - x_tick[..., None, :, :]

        sqdist = np.sum(np.pow(diff, 2), axis=-1)

        out = self.sigma * np.exp(-0.5 * sqdist / (self.ell**2))

        return out
