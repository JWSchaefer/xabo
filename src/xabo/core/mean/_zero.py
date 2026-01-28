import jax.numpy as jnp
from beartype import beartype

from ..typing._typecheck import typecheck
from ._mean import Mean
from ._types import MeanVector, P, S


@beartype
class ZeroMean(Mean[P, S]):
    @typecheck
    def __call__(
        self,
        state: S,
        params: P,
        x: MeanVector,
    ) -> MeanVector:
        return jnp.zeros_like(x)
