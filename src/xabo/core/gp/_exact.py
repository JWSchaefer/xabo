from typing import Any, Optional

import jax.numpy as np
import jax.scipy as scipy
from beartype import beartype
from jaxtyping import Array

from ..kernel._kernel import Kernel
from ..mean._mean import Mean
from ..model._decorators import trained_function, trained_property
from ..model._model import Model
from ..typing import typecheck
from ._types import (
    LowerDecomposition,
    TestFeatures,
    TestPredictions,
    TrainingFeatures,
    TrainingObservations,
)


@beartype
class ExactGP(Model):

    m: Mean
    k: Kernel
    sigma_n: float

    _x: Optional[TrainingFeatures]
    _y: Optional[TrainingObservations]
    _lower: Optional[LowerDecomposition]

    @typecheck
    def __init__(
        self: 'ExactGP',
        mean: Mean,
        kernel: Kernel,
        measurement_noise: float,
    ):
        super().__init__()
        self.m = mean
        self.k = kernel
        self.sigma_n = measurement_noise

        self._x = None
        self._y = None
        self._lower = None
        self._alphas = None

    @typecheck
    def __call__(self: 'ExactGP', x: TestFeatures) -> TestPredictions:
        ...

    @typecheck
    def train(self: 'ExactGP', x: TrainingFeatures, y: TrainingObservations):

        k = self.k(x, x)

        (lower, _) = scipy.linalg.cho_factor(
            k + self.sigma_n * np.eye(k.shape[-1]),
            True,
        )

        alphas = np.linalg.solve(
            lower.T,
            np.linalg.solve(lower, y),
        )

        self._x = x
        self._y = y
        self._lower = lower
        self._alphas = alphas

        self.set_trained()

    @trained_property('lower')
    def lower(self) -> Optional[Array]:
        return self._lower

    @trained_property('y')
    def y(self) -> Optional[Array]:
        return self._y

    @trained_property('x')
    def x(self) -> Optional[Array]:
        return self._x
