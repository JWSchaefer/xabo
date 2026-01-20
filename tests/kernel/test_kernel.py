import jax.numpy as np
import pytest

from ..shared import kernel_id, kernels
from ._cases import KernelTestCase, case_id, cases
from ._decorators import raises


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
