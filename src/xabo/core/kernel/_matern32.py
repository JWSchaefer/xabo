from typing import Generic

import jax.numpy as jnp
from beartype import beartype

from xabo.core.spec._parameter import Parameter

from ..typing._typecheck import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, L


@beartype
class Matern32(Kernel, Generic[L]):

    ell: L
    sigma: Parameter[float]

    @typecheck
    def __call__(
        self,
        state: 'Matern32.State',
        params: 'Matern32.Params',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        ell = params.ell
        sigma = params.sigma

        d = jnp.sum(
            jnp.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (
            (sigma**2)
            * (1 + (jnp.sqrt(3) * d / ell))
            * jnp.exp(-((jnp.sqrt(3) * d / ell)))
        )
