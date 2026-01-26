from typing import Type, TypeVar, get_args, get_origin, get_type_hints

from ._parameter import Parameter


class Spec:
    @classmethod
    def to_param_structure(cls, *, rng=None):
        return {
            name: cls._param_from_annotation(ann)
            for name, ann in get_type_hints(cls).items()
        }

    @classmethod
    def to_param_structure_from_args(cls, args):
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
            name: cls._param_from_annotation(ann)
            for name, ann in resolved_hints.items()
        }

    @classmethod
    def _param_from_annotation(cls, ann):
        origin = get_origin(ann) or ann
        args = get_args(ann)

        if isinstance(origin, type) and issubclass(origin, Parameter):
            return args

        if isinstance(origin, type) and issubclass(origin, Spec):
            return origin.to_param_structure_from_args(args)

        raise TypeError(f'Unsupported annotation {ann}')
