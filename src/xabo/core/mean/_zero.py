from typing import Self

import jax.numpy as np

from xabo.core.mean._mean import Mean
from xabo.core.mean._types import MeanVector


class ZeroMean(Mean):
    def __call__(self: Self, x: MeanVector) -> MeanVector:
        return np.zeros_like(x)
