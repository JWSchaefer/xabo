from typing import TypeVar

from ._parameter import Default

T = TypeVar('T')


class State(Default[T]):
    pass
