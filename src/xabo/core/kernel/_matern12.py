from typing import Generic

import jax.numpy as jnp
from beartype import beartype
from trellis.typing import typecheck

from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, L


@beartype
class Matern12(Kernel, Generic[L]):
    ell: L
    sigma: L

    @typecheck
    def __call__(
        self,
        state: 'Matern12.State',
        params: 'Matern12.Params',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        ell = params.ell.value
        sigma = params.sigma.value

        d = jnp.sum(
            jnp.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (sigma**2) * jnp.exp(-d / ell)
