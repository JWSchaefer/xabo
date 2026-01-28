from typing import Generic

import jax.numpy as jnp
from beartype import beartype

from xabo.core.spec._parameter import Parameter

from ..typing._typecheck import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, R, S


@beartype
class SquaredExponential(Kernel, Generic[R, S]):

    ell: Parameter[R]
    sigma: Parameter[S]

    @typecheck
    def __call__(
        self,
        state: 'SquaredExponential.State',
        params: 'SquaredExponential.Params',
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:

        ell = params.ell
        sigma = params.sigma

        diff = x[..., :, None, :] - x_tick[..., None, :, :]

        sqdist = jnp.sum(jnp.pow(diff / ell, 2), axis=-1)

        out = sigma * jnp.exp(-0.5 * sqdist)

        return out
