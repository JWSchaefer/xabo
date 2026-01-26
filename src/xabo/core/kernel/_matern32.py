# from typing import Generic, TypeVar
#
# import jax.numpy as np
# from beartype import beartype
# from jax import Array
# from jaxtyping import Float
#
# from xabo.core.typing import typecheck
#
# from ._kernel import Kernel
# from ._types import KernelInputA, KernelInputB, KernelOutput, R, S
#
#
# @beartype
# class Matern32(Kernel, Generic[R, S]):
#
#     rho: Parameter[R]
#     sigma: Parameter[S]
#
#     def __init__(self, lengthscale: R, variance: S) -> None:
#         self.rho = lengthscale
#         self.sigma = variance
#
#     @typecheck
#     def __call__(
#         self: 'Matern32',
#         x: KernelInputA,
#         x_tick: KernelInputB,
#     ) -> KernelOutput:
#
#         d = np.sum(
#             np.abs(x[..., :, None, :] - x_tick[..., None, :, :]),
#             axis=-1,
#         )
#
#         return (
#             (self.sigma**2)
#             * (1 + (np.sqrt(3) * d / self.rho))
#             * np.exp(-((np.sqrt(3) * d / self.rho)))
#         )
