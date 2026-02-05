from functools import wraps
from typing import Callable, Optional, Type, TypeVar

import pytest

E = TypeVar("E", bound=Exception)


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
