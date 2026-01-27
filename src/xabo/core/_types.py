from typing import TypeAlias

from jax import Array
from jaxtyping import Float

Scalar: TypeAlias = Float[Array, '']
