from re import S
from typing import Generic

import jax.numpy as jnp
from beartype import beartype

from xabo.core.spec._parameter import Parameter

from ..typing._typecheck import typecheck
from ._kernel import Kernel
from ._types import KernelInputA, KernelInputB, KernelOutput, R, S


@beartype
class Matern12(Kernel, Generic[R, S]):
    """Matern 1/2 kernel (exponential kernel).

    k(x, x') = sigma^2 * exp(-d / rho)

    where d is the L1 distance between x and x'.

    Parameters:
        rho: Lengthscale (positive)
        sigma: Output scale (positive)
    """

    rho: Parameter[R]
    sigma: Parameter[S]

    @typecheck
    def __call__(
        self: 'Matern12',
        state: dict,
        params: dict,
        x: KernelInputA,
        x_tick: KernelInputB,
    ) -> KernelOutput:
        """Evaluate kernel.

        Args:
            params: {'rho': float, 'sigma': float}
            state: State dict (available for non-stationary kernels)
            x: Input array [..., A, X]
            x_tick: Input array [..., B, X]

        Returns:
            Kernel matrix [..., A, B]
        """
        rho = params['rho']
        sigma = params['sigma']

        d = jnp.sum(
            jnp.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
            axis=-1,
        )

        return (sigma**2) * jnp.exp(-d / rho)
