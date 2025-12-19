import jax.numpy as np
import pytest
from jaxtyping import TypeCheckError

from xabo.core.kernel import Matern12, Matern32, Matern52, SquaredExponential

from .case import KernelTestCase, case_id, kernel_id, raises

kernels = [
    Matern12(1.0, 1.0),
    Matern32(1.0, 1.0),
    Matern52(1.0, 1.0),
    SquaredExponential(1.0, 1.0),
]

cases = [
    # No feature dimensions
    KernelTestCase(
        shape_one=(4,),
        shape_two=(6,),
        shape_out=(4, 6),
        exception=TypeCheckError,
    ),
    # Simplest batching
    KernelTestCase(
        shape_one=(4, 1),
        shape_two=(6, 1),
        shape_out=(4, 6),
        exception=None,
    ),
    # Incorrect batching
    KernelTestCase(
        shape_one=(4, 1),
        shape_two=(6, 1),
        shape_out=(6, 4),
        exception=AssertionError,
    ),
    # Shape two too dimensional
    KernelTestCase(
        shape_one=(4, 1),
        shape_two=(6, 1, 1),
        shape_out=(4, 6),
        exception=TypeCheckError,
    ),
    # Two dims of batching
    KernelTestCase(
        shape_one=(2, 7, 4),
        shape_two=(2, 3, 4),
        shape_out=(2, 7, 3),
        exception=None,
    ),
    # Incorrect batching
    KernelTestCase(
        shape_one=(2, 7, 4),
        shape_two=(2, 3, 4),
        shape_out=(2, 3, 7),
        exception=AssertionError,
    ),
    # Mismatched upper dims
    KernelTestCase(
        shape_one=(2, 7, 4),
        shape_two=(1, 3, 4),
        shape_out=(2, 7, 3),
        exception=TypeCheckError,
    ),
    # Three dims of batching
    KernelTestCase(
        shape_one=(4, 2, 7, 4),
        shape_two=(4, 2, 3, 4),
        shape_out=(4, 2, 7, 3),
        exception=None,
    ),
    # Incorrect batching - lower dims
    KernelTestCase(
        shape_one=(4, 2, 7, 4),
        shape_two=(4, 2, 3, 4),
        shape_out=(4, 2, 3, 7),
        exception=AssertionError,
    ),
    # Incorrect batching - upper dims
    KernelTestCase(
        shape_one=(4, 2, 7, 4),
        shape_two=(4, 2, 3, 4),
        shape_out=(2, 4, 7, 3),
        exception=AssertionError,
    ),
    # Mismatched upper dims
    KernelTestCase(
        shape_one=(4, 2, 7, 4),
        shape_two=(3, 2, 3, 4),
        shape_out=(4, 2, 7, 3),
        exception=TypeCheckError,
    ),
]


@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
@pytest.mark.parametrize('case', cases, ids=case_id)
def test_kernel_shapes(kernel, case: KernelTestCase):
    @raises(case.exception)
    def _inner():

        x = np.zeros(case.shape_one)
        y = np.zeros(case.shape_two)

        out = kernel(x, y)
        assert out.shape == case.shape_out

    _inner()
