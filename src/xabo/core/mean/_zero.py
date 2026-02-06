import jax.numpy as jnp
from beartype import beartype
from trellis.typing import typecheck

from ._mean import Mean, P, S, Tr
from ._types import MeanVector


@beartype
class ZeroMean(Mean[P, S, Tr]):
    @typecheck
    def __call__(
        self,
        state: S,
        params: P,
        x: MeanVector,
    ) -> MeanVector:
        return jnp.zeros_like(x)
