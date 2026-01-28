from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

from jaxtyping import TypeCheckError

Shape = Tuple[int, ...]
Outcome = Union[Shape, Type[Exception]]


@dataclass(frozen=True)
class KernelTestCase:
    shape_one: Shape
    shape_two: Shape
    shape_out: Shape
    exception: Optional[Type[Exception]]


def case_id(case: 'KernelTestCase'):
    return f'{case.shape_one}×{case.shape_two}→{case.shape_out}'


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
        shape_one=(4, 2, 7, 1),
        shape_two=(4, 2, 3, 1),
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
