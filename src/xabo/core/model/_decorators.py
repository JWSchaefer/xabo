from functools import wraps

from beartype.typing import Callable, Concatenate, ParamSpec, TypeVar

from ._model import Model

T = TypeVar('T')
P = ParamSpec('P')
M = TypeVar('M', bound=Model)


def trained_property(name: str):
    N = TypeVar('N', bound=Model)

    def decorator(method: Callable[[N], T]) -> property:
        private_name = '_' + name

        @wraps(method)
        def getter(self: N) -> T:
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


def trained_function(
    method: Callable[Concatenate[M, P], T]
) -> Callable[Concatenate[M, P], T]:
    @wraps(method)
    def wrapper(self: M, *args: P.args, **kwargs: P.kwargs) -> T:
        if not self.is_trained:
            raise AttributeError(
                'Attempted to call training-dependent function '
                'before training. Try training the model first.'
            )

        return method(self, *args, **kwargs)

    return wrapper
