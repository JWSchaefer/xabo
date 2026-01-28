from xabo.core.kernel import (
    Kernel,
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
)

kernels: list[Kernel] = [
    Matern12(ell=0.1, sigma=0.1),
    Matern32(ell=0.1, sigma=0.1),
    Matern52(ell=0.1, sigma=0.1),
    SquaredExponential(ell=0.1, sigma=0.1),
]


def kernel_id(kernel: Kernel) -> str:
    return type(kernel).__name__
