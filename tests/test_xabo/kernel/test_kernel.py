from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.util import Array
from trellis import Model

from xabo.core.kernel import Kernel

from ..shared import kernel_id, kernels
from ._cases import KernelTestCase, case_id, cases
from ._decorators import raises


@pytest.mark.parametrize('kernel_spec', kernels, ids=kernel_id)
@pytest.mark.parametrize('case', cases, ids=case_id)
def test_kernel_shapes(
    kernel_spec: Kernel, case: KernelTestCase, kernel_fixtures
):
    @raises(case.exception)
    def _inner():
        kernel = Model.from_spec(kernel_spec)

        x: Array
        y: Array

        expected: Optional[Array] = None

        if case.exception is None:
            fixture_key = f'{kernel_id(kernel_spec)}_{case_id(case)}'
            x = kernel_fixtures[f'{fixture_key}_x']
            y = kernel_fixtures[f'{fixture_key}_y']
            expected = kernel_fixtures[f'{fixture_key}_out']

        else:
            x = jnp.zeros(case.shape_one)
            y = jnp.zeros(case.shape_two)

        out = kernel(x, y)
        assert out.shape == case.shape_out
        if expected is not None:
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-15)

    _inner()
