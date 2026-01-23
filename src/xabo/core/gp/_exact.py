import jax.numpy as np
import jax.scipy as scipy
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float

from ..kernel._kernel import Kernel
from ..mean._mean import Mean
from ..model._decorators import trained_function, trained_property
from ..model._model import Model
from ..typing import typecheck
from ._types import (
    LowerDecomposition,
    TestFeatures,
    TestPredictions,
    TestVariance,
    TrainingFeatures,
    TrainingObservations,
)


@beartype
class ExactGP(Model):

    m: Mean
    k: Kernel
    sigma_n: float | np.ndarray

    _x: Optional[TrainingFeatures]
    _y: Optional[TrainingObservations]

    _lower: Optional[LowerDecomposition]
    _alphas: Optional[TrainingObservations]

    @typecheck
    def __init__(
        self: 'ExactGP',
        mean: Mean,
        kernel: Kernel,
        measurement_noise: float | np.ndarray,
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
    @trained_function
    def __call__(
        self: 'ExactGP', x_test: TestFeatures
    ) -> Tuple[TestPredictions, TestVariance]:

        k_star = self.k(self.x, x_test)
        mu = k_star.T @ self.alphas
        v = scipy.linalg.solve(self.lower, k_star, lower=True)
        var_diag = np.diag(self.k(x_test, x_test)) - np.sum(v**2, axis=0)
        return (mu, var_diag[:, None])

    @typecheck
    def train(
        self: 'ExactGP',
        x: TrainingFeatures,
        y: TrainingObservations,
        jitter: Optional[float] = None,
    ):
        if jitter is None:
            if x.dtype == np.float64:
                jitter = 1e-12
            elif x.dtype == np.float32:
                jitter = 1e-7
            elif x.dtype == np.float16:
                jitter = 1e-4
            else:
                raise ValueError(
                    f'Unable to automatically assign jitter for {x.dtype}. Please provide an value explicitly using the `jitter : float` kwarg.',
                )

        k = self.k(x, x) + (self.sigma_n**2 + jitter) * np.eye(x.shape[-2])

        lower = scipy.linalg.cholesky(k, lower=True)

        alphas = scipy.linalg.solve(
            lower.T,
            scipy.linalg.solve(lower, y, lower=True),
            lower=False,
        )

        self._x = x
        self._y = y
        self._lower = lower
        self._alphas = alphas

        self.set_trained()

    @typecheck
    @trained_function
    def log_marginal_liklihood(self: 'ExactGP') -> Float[Array, '']:
        return (
            -(1 / 2) * self.y.T @ self.alphas
            - np.sum(np.log(np.diag(self.lower)))
            - (self.lower.shape[-1] / 2) * np.log(2 * np.pi)
        ).squeeze()

    @trained_property('y')
    def y(self) -> Optional[Array]:
        return self._y

    @trained_property('x')
    def x(self) -> Optional[Array]:
        return self._x

    @trained_property('lower')
    def lower(self) -> Optional[Array]:
        return self._lower

    @trained_property('alphas')
    def alphas(self) -> Optional[Array]:
        return self._alphas
