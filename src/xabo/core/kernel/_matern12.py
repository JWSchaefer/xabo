from typing import Generic

import jax.numpy as np
from beartype import beartype
from jaxtyping import PyTree

from xabo.core.spec._parameter import Parameter
from xabo.core.typing import typecheck

from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, R, S


@beartype
class Matern12(Kernel, Generic[R, S]):

    rho: R
    sigma: S

    @typecheck
    def __call__(
        self: 'Matern12',
        state: PyTree,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        sigma = state['sigma']
        rho = state['rho']

        d = np.sum(
            np.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (sigma**2) * np.exp(-d / rho)
