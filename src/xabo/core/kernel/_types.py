from typing import TypeAlias

from jaxtyping import Array, Float

KernelInputA: TypeAlias = Float[Array, '*S A X']
KernelInputB: TypeAlias = Float[Array, '*S B X']
KernelOutput: TypeAlias = Float[Array, '*S A B']
