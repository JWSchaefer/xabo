import warnings
from abc import ABC
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
from ._transform import Log


class Spec(ABC):
    """Base class for declarative model definitions.

    Subclasses define model structure via type hints:

        class MyModel(Spec):
            kernel: Matern12[Float[Array, '...'], float]
            x: State[Float[Array, '*S N X']]
            observation_noise: Parameter[float]

    Instantiate with parameter values:

        model = MyModel(
            kernel=Matern12(rho=0.5, sigma=1.0),
            observation_noise=0.01,
        )

    Extract pytrees:

        params = model.init_params()
        state = model.init_state()
    """

    def __init__(self, **kwargs):
        """Initialize with values for Parameters and nested Specs.

        Args:
            **kwargs: Values for each Parameter and Spec field defined in type hints.
        """
        hints = get_type_hints(self.__class__)

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif isinstance(origin, type) and issubclass(origin, State):
                setattr(self, name, None)
            else:
                raise TypeError(f'Missing required argument: {name}')

    def init_params(
        self, rng: Optional[Array] = None, from_prior: bool = False
    ) -> dict:
        """Extract parameter values into a params pytree.

        Args:
            rng: Random key for sampling from priors (required if from_prior=True)
            from_prior: If True, sample from priors instead of using instance values

        Returns:
            Nested dict of parameter values matching the model structure.
        """
        if from_prior and rng is None:
            raise ValueError('rng required when from_prior=True')

        hints = get_type_hints(self.__class__)
        params = {}

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                value = getattr(self, name)
                if from_prior:
                    raise NotImplementedError(
                        'Sampling from priors not yet implemented'
                    )
                params[name] = value

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                if from_prior:
                    params[name] = nested_spec.init_params(
                        rng, from_prior=True
                    )
                else:
                    params[name] = nested_spec.init_params()

        return params

    def init_state(self) -> dict:
        """Generate state pytree with None placeholders.

        Returns:
            Nested dict with None for each State field.
        """
        hints = get_type_hints(self.__class__)
        state = {}

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                state[name] = None

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_state = nested_spec.init_state()
                if nested_state:
                    state[name] = nested_state

        return state

    def _get_transforms(self) -> dict:
        """Get transforms for each parameter.

        Returns:
            Nested dict mapping param names to their Transform instances.
        """
        hints = get_type_hints(self.__class__)
        transforms = {}

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                transforms[name] = getattr(origin, 'transform', Log())

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_transforms = nested_spec._get_transforms()
                if nested_transforms:
                    transforms[name] = nested_transforms

        return transforms

    def _get_priors(self) -> dict:
        """Get priors for each parameter.

        Returns:
            Nested dict mapping param names to their Prior instances (or None).
        """
        hints = get_type_hints(self.__class__)
        priors = {}

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                prior = getattr(origin, 'prior', None)
                if prior is not None:
                    priors[name] = prior

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_priors = nested_spec._get_priors()
                if nested_priors:
                    priors[name] = nested_priors

        return priors

    @classmethod
    def to_spec(cls: Type['Spec']) -> dict:
        """Extract type specification from class annotations.

        Returns:
            Dict mapping field names to their type markers (Parameter/State/nested Spec).
        """
        return {
            name: cls._from_annotation(ann)
            for name, ann in get_type_hints(cls).items()
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
