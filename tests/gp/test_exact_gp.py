import jax.numpy as np
import pytest

from xabo.core.gp import ExactGP
from xabo.core.kernel import Kernel
from xabo.core.mean import Mean

from ..shared import kernel_id, kernels, mean_id, means


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_constructor(mean: Mean, kernel: Kernel):
    _ = ExactGP(mean, kernel, 1e-6)


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_train(mean: Mean, kernel: Kernel):
    gp = ExactGP(mean, kernel, 1e-3)
    x = np.linspace(0, 1, 100)[:, None]
    y = np.zeros_like(x)
    gp.train(x, y)


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_access_correct(mean: Mean, kernel: Kernel):
    gp = ExactGP(mean, kernel, 1e-6)
    x = np.linspace(0, 1, 100)[:, None]
    y = np.zeros_like(x)
    gp.train(x, y)
    _ = gp.x
    _ = gp.y
    _ = gp.lower


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_access_incorrect(mean: Mean, kernel: Kernel):
    gp = ExactGP(mean, kernel, 1e-6)
    with pytest.raises(AttributeError):
        _ = gp.x
    with pytest.raises(AttributeError):
        _ = gp.y
    with pytest.raises(AttributeError):
        _ = gp.lower
    with pytest.raises(AttributeError):
        _ = gp.alphas


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_call_correct(mean: Mean, kernel: Kernel):
    gp = ExactGP(mean, kernel, 1e-6)
    x = np.linspace(0, 1, 100)[:, None]
    y = np.zeros_like(x)
    gp.train(x, y)
    gp(x[:4])


@pytest.mark.parametrize('mean', means, ids=mean_id)
@pytest.mark.parametrize('kernel', kernels, ids=kernel_id)
def test_call_incorrect(mean: Mean, kernel: Kernel):
    gp = ExactGP(mean, kernel, 1e-6)
    x = np.linspace(0, 1, 100)[:, None]
    with pytest.raises(AttributeError):
        gp(x)
