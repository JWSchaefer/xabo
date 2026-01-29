from typing import Any, TypeAlias, TypeVar

from jax import Array
from jaxtyping import Float

from ..spec._parameter import Parameter

KernelInputA: TypeAlias = Float[Array, '*S A X']
KernelInputB: TypeAlias = Float[Array, '*S B X']
KernelOutput: TypeAlias = Float[Array, '*S A B']

L = TypeVar('L', bound=Parameter[Any])
