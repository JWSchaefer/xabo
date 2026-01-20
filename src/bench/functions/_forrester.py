from typing import TypeAlias

import jax.numpy as np
from jaxtyping import Array, Float

KernelInputA: TypeAlias = Float[Array, '*S A X']
KernelInputB: TypeAlias = Float[Array, '*S B X']
KernelOutput: TypeAlias = Float[Array, '*S A B']


def _f(x: float):
    return np.pow((6 * x - 2), 2) * np.sin(12 * x - 4)


def forrester(x: float, a: float = 1.0, b: float = 0.0, c: float = 0.0):
    return a * _f(x) + b * (x - 0.5) - c
