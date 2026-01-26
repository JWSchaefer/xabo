from typing import Self, Type, get_origin, get_type_hints

from ._parameter import Parameter
from ._state import State
from ._tree import Tree


class Spec:
    @classmethod
    def to_param_structure(cls: Type[Self], *, rng=None):

        params = {
            name: print(ty)
            for (name, ty) in get_type_hints(cls).items()
            if issubclass(get_origin(ty), Spec)
        }

        params = {
            name: ty.to_param_structure()
            for (name, ty) in get_type_hints(cls).items()
            if issubclass(get_origin(ty) or ty, Spec)
        }

        print(params)

        return params
