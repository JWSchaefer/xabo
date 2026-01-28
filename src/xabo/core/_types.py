from typing import TypeAlias, TypeVar

from jaxtyping import Array, Float

T = TypeVar('T', float, Float[Array, ''], Float[Array, '...'])

Scalar: TypeAlias = Float[Array, ''] | Float[Array, 'N']
