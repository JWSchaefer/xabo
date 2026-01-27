from typing import TypeAlias, TypeVar

from jax import Array
from jaxtyping import Float

KernelInputA: TypeAlias = Float[Array, '*S A X']
KernelInputB: TypeAlias = Float[Array, '*S B X']
KernelOutput: TypeAlias = Float[Array, '*S A B']

R = TypeVar('R', float, Array)
S = TypeVar('S', float, Array)
