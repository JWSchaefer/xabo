import jax.numpy as np

from xabo.core.kernel import SquaredExponential


def test_two_dim():
    x = np.zeros((7, 4))
    x_tick = np.zeros((3, 4))

    kernel = SquaredExponential(1.0, 1.0)

    k = kernel(x, x_tick)

    assert k.shape == (7, 3)


def test_thee_dim():
    x = np.zeros((2, 7, 4))
    x_tick = np.zeros((2, 3, 4))

    kernel = SquaredExponential(1.0, 1.0)

    k = kernel(x, x_tick)

    assert k.shape == (2, 7, 3)
