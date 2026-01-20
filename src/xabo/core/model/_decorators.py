from functools import wraps
from typing import Callable, TypeVar

from xabo.core.model._model import Model

S = TypeVar('S', bound=Model)
T = TypeVar('T')


def trained_property(
    name: str,
) -> Callable[[Callable[[S], T]], property]:
    def decorator(method: Callable[[S], T]) -> property:
        private_name = '_' + name

        @wraps(method)
        def getter(self: S) -> T:
            if not self.is_trained:
                raise AttributeError(
                    f'Attempted to access training-dependent variable `{name}` '
                    'before training. Try training the model first.'
                )

            value = getattr(self, private_name, None)
            if value is None:
                raise AttributeError(
                    f'Attempted to access training-dependent variable `{name}` '
                    'but it was None. Did you train the model first?'
                )

            return method(self)

        return property(getter)

    return decorator


def trained_function(method: Callable[[S], T]) -> Callable[[S], T]:
    @wraps(method)
    def wrapper(self: S) -> T:

        if not self.is_trained:
            raise AttributeError(
                'Attempted to call training-dependent function. '
                + 'before training. Try training the model first.'
            )

        return method(self)

    return wrapper
