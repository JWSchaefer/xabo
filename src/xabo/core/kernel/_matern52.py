from typing import Generic

import jax.numpy as jnp
from beartype import beartype

from xabo.core.spec._parameter import Parameter

from ..typing._typecheck import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, R, S


@beartype
class Matern52(Kernel, Generic[R, S]):

    rho: Parameter[R]
    sigma: Parameter[S]

    @typecheck
    def __call__(
        self,
        state: 'Matern52.State',
        params: 'Matern52.Params',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        rho = params.rho
        sigma = params.sigma

        d = jnp.sum(
            jnp.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (
            (sigma**2)
            * (
                1
                + (jnp.sqrt(5) * d / rho)
                + (5 * jnp.pow(d, 2) / (3 * (rho**2)))
            )
            * jnp.exp(-((jnp.sqrt(5) * d / rho)))
        )
