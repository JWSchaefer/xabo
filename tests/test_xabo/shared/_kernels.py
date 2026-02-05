from xabo.core.kernel import (
    Kernel,
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
)
from xabo.core.prior import NoPrior

kernels: list[Kernel] = [
    Matern12(ell=NoPrior(value=0.1), sigma=NoPrior(value=0.1)),
    Matern32(ell=NoPrior(value=0.1), sigma=NoPrior(value=0.1)),
    Matern52(ell=NoPrior(value=0.1), sigma=NoPrior(value=0.1)),
    SquaredExponential(ell=NoPrior(value=0.1), sigma=NoPrior(value=0.1)),
]


def kernel_id(kernel: Kernel) -> str:
    return type(kernel).__name__
