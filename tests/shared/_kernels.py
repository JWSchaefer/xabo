from xabo.core.kernel import (
    Kernel,
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
)

kernels = [
    Matern12(1.0, 1.0),
    Matern32(1.0, 1.0),
    Matern52(1.0, 1.0),
    SquaredExponential(1.0, 1.0),
]


def kernel_id(kernel: Kernel) -> str:
    return type(kernel).__name__
