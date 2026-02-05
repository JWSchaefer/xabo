from __future__ import annotations

from typing import Any, TypeAlias, TypeVar

from jax import Array
from jaxtyping import Float

from ..prior._prior import Prior

KernelInputA: TypeAlias = Float[Array, "*S A X"]
KernelInputB: TypeAlias = Float[Array, "*S B X"]
KernelOutput: TypeAlias = Float[Array, "*S A B"]

# TypeVar bounded to Prior - all kernel parameters must be wrapped in a Prior
L = TypeVar("L", bound=Prior[Any])
