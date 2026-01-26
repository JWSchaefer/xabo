from typing import (
    Optional,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from jax import Array

from ._parameter import Parameter
from ._state import State


class Spec:
    @classmethod
    def to_spec(cls: Type['Spec']) -> dict:
        return {
            name: cls._from_annotation(ann)
            for name, ann in get_type_hints(cls).items()
        }

    @classmethod
    def to_params(cls: Type['Spec'], rng: Optional[Array]):
        return {i for i in cls.to_spec()}

    @classmethod
    def to_structure_from_args(cls, args):
        type_vars = getattr(cls, '__parameters__', ())
        tv_map: dict[TypeVar, Type] = dict(zip(type_vars, args))

        def substitute(ann):
            if isinstance(ann, TypeVar):
                return tv_map[ann]

            origin = get_origin(ann)
            if origin is None:
                return ann

            return origin[tuple(substitute(a) for a in get_args(ann))]

        resolved_hints = {
            name: substitute(ann) for name, ann in get_type_hints(cls).items()
        }

        return {
            name: cls._from_annotation(ann)
            for name, ann in resolved_hints.items()
        }

    @classmethod
    def _from_annotation(cls, ann):
        origin = get_origin(ann) or ann
        args = get_args(ann)

        if isinstance(origin, type) and issubclass(origin, (Parameter, State)):
            return origin

        if isinstance(origin, type) and issubclass(origin, Spec):
            return origin.to_structure_from_args(args)

        raise TypeError(f'Unsupported annotation {ann}')
