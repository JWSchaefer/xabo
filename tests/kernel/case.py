from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

import pytest

Shape = Tuple[int, ...]
Outcome = Union[Shape, Type[Exception]]
E = TypeVar('E', bound=Exception)


def kernel_id(kernel):
    return type(kernel).__name__


def case_id(case: 'KernelTestCase'):
    return f'{case.shape_one}×{case.shape_two}→{case.shape_out}'


def raises(exc: Optional[Type[E]]):
    def decorator(func: Callable):
        if exc is None:
            return func

        expected_exc: Type[E] = exc

        @wraps(func)
        def wrapper(*args, **kwargs):
            with pytest.raises(expected_exc):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass(frozen=True)
class KernelTestCase:
    shape_one: Shape
    shape_two: Shape
    shape_out: Shape
    exception: Optional[Type[Exception]]
