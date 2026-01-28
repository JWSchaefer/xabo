from typing import TypeAlias, TypeVar

from jaxtyping import Array, Float

P = TypeVar('P')
S = TypeVar('S')
MeanVector: TypeAlias = Float[Array, '*S X']
