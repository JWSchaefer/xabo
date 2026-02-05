from pathlib import Path

import jax.numpy as jnp
from jax import random

from xabo.core import Model

from ..shared import kernel_id, kernels
from ._cases import case_id, cases

SEED = 42


def generate_kernel_fixtures(output_path: Path):
    key = random.PRNGKey(SEED)
    fixtures = {}

    valid_cases = [c for c in cases if c.exception is None]

    for kernel_spec in kernels:
        kernel_name = kernel_id(kernel_spec)
        kernel = Model.from_spec(kernel_spec)

        for case in valid_cases:
            key, subkey1, subkey2 = random.split(key, 3)

            x = random.normal(subkey1, case.shape_one)
            y = random.normal(subkey2, case.shape_two)

            output = kernel(x, y)

            # Use kernel name and case id as fixture key
            fixture_key = f"{kernel_name}_{case_id(case)}"
            fixtures[f"{fixture_key}_x"] = x
            fixtures[f"{fixture_key}_y"] = y
            fixtures[f"{fixture_key}_out"] = output

    jnp.savez(output_path, **fixtures)


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    generate_kernel_fixtures(fixtures_dir / "kernel_expected.npz")
