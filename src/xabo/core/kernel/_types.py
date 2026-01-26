from typing import TypeAlias, TypeVar

from jaxtyping import Array, Float

from xabo.core.spec._parameter import Parameter

KernelInputA: TypeAlias = Float[Array, '*S A X']
KernelInputB: TypeAlias = Float[Array, '*S B X']
KernelOutput: TypeAlias = Float[Array, '*S A B']

R = TypeVar('R', bound=Parameter)
S = TypeVar('S', bound=Parameter)
