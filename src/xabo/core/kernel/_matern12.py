from typing import Generic

import jax.numpy as jnp
from beartype import beartype

from xabo.core.spec._parameter import Parameter

from ..typing._typecheck import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, L


@beartype
class Matern12(Kernel, Generic[L]):

    ell: L
    sigma: Parameter[float]

    @typecheck
    def __call__(
        self,
        state: 'Matern12.State',
        params: 'Matern12.Params',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        ell = params.ell
        sigma = params.sigma

        d = jnp.sum(
            jnp.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (sigma**2) * jnp.exp(-d / ell)
