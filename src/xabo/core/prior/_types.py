from typing import TypeVar

from jaxtyping import Array, Float

T = TypeVar('T', Float[Array, ''], Float[Array, 'N'])
