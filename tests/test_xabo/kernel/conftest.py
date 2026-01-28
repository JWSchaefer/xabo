from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from .generate_fixtures import generate_kernel_fixtures

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
FIXTURE_FILE = FIXTURES_DIR / 'kernel_expected.npz'


def pytest_configure(config):
    """Regenerate fixtures if flag is passed."""
    if config.getoption('--regenerate-fixtures', default=False):
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        generate_kernel_fixtures(FIXTURE_FILE)


@pytest.fixture(scope='session')
def kernel_fixtures():
    """Load kernel expected output fixtures."""
    loaded = np.load(FIXTURE_FILE)
    return {key: jnp.asarray(loaded[key]) for key in loaded.files}
