from abc import ABC, ABCMeta
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import jax.tree_util
from jax import Array

from ..prior import Prior
from ..transform import Log
from ..transform._transform import Transform
from ._parameter import Parameter
from ._state import State

P = TypeVar('P')
S = TypeVar('S')
Pr = TypeVar('Pr')
Tr = TypeVar('Tr')

_params_class_cache: dict[type, type] = {}
_state_class_cache: dict[type, type] = {}
_priors_class_cache: dict[type, type] = {}
_transforms_class_cache: dict[type, type] = {}

_GENERATED_ATTRS = ('Params', 'State', 'Priors', 'Transforms')


def _generate_params_class(spec_cls: type) -> type:
    """Generate frozen dataclass for a Spec's parameters."""
    if spec_cls in _params_class_cache:
        return _params_class_cache[spec_cls]

    hints = get_type_hints(spec_cls)
    fields = []

    for name, ann in hints.items():
        # Skip class attributes set by metaclass and Generic type params
        if name in _GENERATED_ATTRS:
            continue

        origin = get_origin(ann) or ann

        if isinstance(origin, type) and issubclass(origin, Parameter):
            inner_type = get_args(ann)[0] if get_args(ann) else Any
            fields.append((name, inner_type))

        elif isinstance(origin, TypeVar):
            # Handle TypeVars bound to Parameter (including Parameter[Any])
            bound = getattr(origin, '__bound__', None)
            if bound is not None:
                bound_origin = get_origin(bound) or bound
                if isinstance(bound_origin, type) and issubclass(
                    bound_origin, Parameter
                ):
                    fields.append((name, Any))

        elif isinstance(origin, type) and issubclass(origin, Spec):
            # Nested Spec - recursively generate its Params class
            nested_params = _generate_params_class(origin)
            fields.append((name, nested_params))

    class_name = f'{spec_cls.__name__}Params'
    ParamsClass = make_dataclass(class_name, fields, frozen=True)

    # Register as JAX PyTree
    jax.tree_util.register_dataclass(
        ParamsClass,
        data_fields=[f[0] for f in fields],
        meta_fields=[],
    )

    _params_class_cache[spec_cls] = ParamsClass
    return ParamsClass


def _generate_state_class(spec_cls: type) -> type:
    """Generate frozen dataclass for a Spec's state."""
    if spec_cls in _state_class_cache:
        return _state_class_cache[spec_cls]

    hints = get_type_hints(spec_cls)
    fields = []

    for name, ann in hints.items():
        # Skip class attributes set by metaclass
        if name in _GENERATED_ATTRS:
            continue

        origin = get_origin(ann) or ann

        if isinstance(origin, type) and issubclass(origin, State):
            inner_type = get_args(ann)[0] if get_args(ann) else Any
            # State fields default to None
            fields.append((name, Optional[inner_type], None))

        elif isinstance(origin, type) and issubclass(origin, Spec):
            nested_state = _generate_state_class(origin)
            # Only include if nested spec has state fields
            if dataclass_fields(nested_state):
                fields.append((name, nested_state))

    class_name = f'{spec_cls.__name__}State'
    StateClass = make_dataclass(class_name, fields, frozen=True)

    # Register as JAX PyTree
    jax.tree_util.register_dataclass(
        StateClass,
        data_fields=[f[0] for f in fields],
        meta_fields=[],
    )

    _state_class_cache[spec_cls] = StateClass
    return StateClass


def _generate_attribute_class(
    spec_cls: type,
    attr_name: str,
    target_type: type,
    class_suffix: str,
    cache: dict[type, type],
    default_factory: Optional[Callable[[], Any]] = None,
) -> type:
    """Generate frozen dataclass with fields for Parameters that have the attribute.

    Args:
        spec_cls: The Spec class to generate for
        attr_name: Attribute name to extract ('prior' or 'transform')
        target_type: Type for the fields (Prior or Transform)
        class_suffix: Suffix for generated class name ('Priors' or 'Transforms')
        cache: Cache dict to store/retrieve generated classes
        default_factory: Optional factory for default values (e.g., Log for transforms)

    Returns:
        Generated frozen dataclass type.
    """
    if spec_cls in cache:
        return cache[spec_cls]

    hints = get_type_hints(spec_cls)
    fields = []

    for name, ann in hints.items():
        if name in _GENERATED_ATTRS:
            continue

        origin = get_origin(ann) or ann

        if isinstance(origin, type) and issubclass(origin, Parameter):
            attr_value = getattr(origin, attr_name, None)
            if attr_value is not None or default_factory is not None:
                fields.append((name, target_type))

        elif isinstance(origin, TypeVar):
            bound = getattr(origin, '__bound__', None)
            if bound is not None:
                bound_origin = get_origin(bound) or bound
                if isinstance(bound_origin, type) and issubclass(
                    bound_origin, Parameter
                ):
                    # For TypeVars, we include the field - actual type resolved at runtime
                    fields.append((name, target_type))

        elif isinstance(origin, type) and issubclass(origin, Spec):
            nested_cls = _generate_attribute_class(
                origin,
                attr_name,
                target_type,
                class_suffix,
                cache,
                default_factory,
            )
            if dataclass_fields(nested_cls):
                fields.append((name, nested_cls))

    class_name = f'{spec_cls.__name__}{class_suffix}'
    AttrClass = make_dataclass(class_name, fields, frozen=True)

    cache[spec_cls] = AttrClass
    return AttrClass


class SpecMeta(ABCMeta):
    """Metaclass that generates Params, State, Priors, and Transforms classes for Specs."""

    ATTRS = _GENERATED_ATTRS

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Spec class
        if name == 'Spec':
            return cls

        # Generate and attach Params/State classes
        # Use setattr to avoid type checker warnings about dynamic attributes
        setattr(cls, 'Params', _generate_params_class(cls))
        setattr(cls, 'State', _generate_state_class(cls))
        setattr(
            cls,
            'Priors',
            _generate_attribute_class(
                cls, 'prior', Prior, 'Priors', _priors_class_cache
            ),
        )
        setattr(
            cls,
            'Transforms',
            _generate_attribute_class(
                cls,
                'transform',
                Transform,
                'Transforms',
                _transforms_class_cache,
                default_factory=Log,
            ),
        )

        return cls


class Spec(ABC, Generic[P, S, Pr, Tr], metaclass=SpecMeta):
    """Base class for declarative model definitions.

    Subclasses define model structure via type hints. For full type safety,
    annotate with the generated Params/State/Priors/Transforms types:

        class MyModel(Spec['MyModelParams', 'MyModelState', 'MyModelPriors', 'MyModelTransforms']):
            kernel: Matern12[float, float]
            x: State[Float[Array, '*S N X']]
            observation_noise: Parameter[float]

    Instantiate with parameter values:

        model = MyModel(
            kernel=Matern12(rho=0.5, sigma=1.0),
            observation_noise=0.01,
        )

    Extract typed pytrees with named attribute access:

        params = model.init_params()  # MyModelParams
        params.kernel.rho             # 0.5
    """

    Params: type[P]  # Set by metaclass
    State: type[S]  # Set by metaclass
    Priors: type[Pr]  # Set by metaclass
    Transforms: type[Tr]  # Set by metaclass

    def __init__(self, **kwargs):
        """Initialize with values for Parameters and nested Specs.

        Args:
            **kwargs: Values for each Parameter and Spec field defined in type hints.

        Raises:
            TypeError: If required arguments are missing or unexpected arguments provided.
        """
        hints = get_type_hints(self.__class__)
        expected = {n for n in hints if n not in SpecMeta.ATTRS}

        # Check for unexpected arguments
        unexpected = set(kwargs) - expected
        if unexpected:
            raise TypeError(
                f"Unexpected argument(s): {', '.join(sorted(unexpected))}"
            )

        for name, ann in hints.items():
            # Skip class attributes set by metaclass
            if name in SpecMeta.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif isinstance(origin, type) and issubclass(origin, State):
                setattr(self, name, None)
            else:
                raise TypeError(f'Missing required argument: {name}')

    def init_params(self, rng: Optional[Array] = None) -> P:
        """Extract parameter values into a typed Params pytree.

        Args:
            rng: Random key for sampling from priors (required if from_prior=True)
            from_prior: If True, sample from priors instead of using instance values

        Returns:
            Typed Params dataclass with named attribute access.
        """
        params_dict = self._collect_params_dict(rng)
        return self._dict_to_params(params_dict)

    def _collect_params_dict(
        self, rng: Optional[Array] = None, from_prior: bool = False
    ) -> dict:
        """Collect parameter values into a dict."""
        if from_prior and rng is None:
            raise ValueError('rng required when from_prior=True')

        hints = get_type_hints(self.__class__)
        params = {}

        for name, ann in hints.items():
            # Skip class attributes
            if name in _GENERATED_ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                value = getattr(self, name)
                if from_prior:
                    raise NotImplementedError(
                        'Sampling from priors not yet implemented'
                    )
                params[name] = value

            elif isinstance(origin, TypeVar):
                bound = getattr(origin, '__bound__', None)
                if bound is not None:
                    bound_origin = get_origin(bound) or bound
                    if isinstance(bound_origin, type) and issubclass(
                        bound_origin, Parameter
                    ):
                        value = getattr(self, name)
                        if from_prior:
                            raise NotImplementedError(
                                'Sampling from priors not yet implemented'
                            )
                        params[name] = value

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                if from_prior:
                    params[name] = nested_spec._collect_params_dict(
                        rng, from_prior=True
                    )
                else:
                    params[name] = nested_spec._collect_params_dict()

        return params

    def _dict_to_params(self, params_dict: dict) -> P:
        """Convert nested dict to typed Params dataclass."""
        hints = get_type_hints(self.__class__)
        converted = {}

        for name, value in params_dict.items():
            ann = hints.get(name)
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                converted[name] = nested_spec._dict_to_params(value)
            else:
                converted[name] = value

        return self.__class__.Params(**converted)

    def init_state(self) -> S:
        """Generate typed State pytree with None placeholders.

        Returns:
            Typed State dataclass with named attribute access.
        """
        state_dict = self._collect_state_dict()
        return self._dict_to_state(state_dict)

    def _collect_state_dict(self) -> dict:
        """Collect state values into a dict."""
        hints = get_type_hints(self.__class__)
        state = {}

        for name, ann in hints.items():
            # Skip class attributes
            if name in _GENERATED_ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                state[name] = None

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_state = nested_spec._collect_state_dict()
                if nested_state:
                    state[name] = nested_state

        return state

    def _dict_to_state(self, state_dict: dict) -> S:
        """Convert nested dict to typed State dataclass."""
        hints = get_type_hints(self.__class__)
        converted = {}

        for name, value in state_dict.items():
            ann = hints.get(name)
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                converted[name] = nested_spec._dict_to_state(value)
            else:
                converted[name] = value

        return self.__class__.State(**converted)

    def _collect(
        self,
        extractor: Callable[[type], Any],
        include_none: bool = False,
    ) -> dict:
        """Collect values from Parameter leaves using an extractor function.

        Args:
            extractor: Function that takes a Parameter class and returns a value
            include_none: Whether to include None values

        Returns:
            Nested dict mapping param names to extracted values.
        """
        hints = get_type_hints(self.__class__)
        result = {}

        # Build TypeVar -> actual type mapping from __orig_class__
        typevar_map: dict[TypeVar, type] = {}
        orig_class = getattr(self, '__orig_class__', None)
        if orig_class is not None:
            type_params = getattr(self.__class__, '__parameters__', ())
            type_args = get_args(orig_class)
            typevar_map = dict(zip(type_params, type_args))

        for name, ann in hints.items():
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                value = extractor(origin)
                if include_none or value is not None:
                    result[name] = value

            elif isinstance(origin, TypeVar):
                # Resolve TypeVar to actual type from generic parameterization
                actual_type = typevar_map.get(origin)
                if actual_type is not None and isinstance(actual_type, type):
                    if issubclass(actual_type, Parameter):
                        value = extractor(actual_type)
                        if include_none or value is not None:
                            result[name] = value

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_result = nested_spec._collect(extractor, include_none)
                if nested_result:
                    result[name] = nested_result

        return result

    def _collect_by_type(
        self,
        target_type: type,
        include_none: bool = False,
    ) -> dict:
        """Collect attributes matching a type from Parameter leaves.

        Args:
            target_type: The type to search for (e.g., Transform, Prior)
            include_none: Whether to include None values

        Returns:
            Nested dict mapping param names to matching attribute values.
        """

        def extractor(param_cls: type) -> Any:
            for attr_name in dir(param_cls):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(param_cls, attr_name, None)
                if isinstance(attr, target_type):
                    return attr
            return None

        return self._collect(extractor, include_none)

    def get_priors(self) -> Pr:
        """Get priors with named attribute access.

        Returns:
            Priors dataclass with named attributes for each parameter that has a prior.
        """
        return self._build_attribute_instance('Priors', 'prior', None)

    def get_transforms(self) -> Tr:
        """Get transforms with named attribute access.

        Returns:
            Transforms dataclass with named attributes for each parameter.
        """
        return self._build_attribute_instance('Transforms', 'transform', Log)

    def _build_attribute_instance(
        self, class_attr: str, attr_name: str, default_factory: Optional[type]
    ) -> Any:
        """Build an instance of a generated attribute class.

        Args:
            class_attr: Name of the class attribute ('Priors' or 'Transforms')
            attr_name: Attribute to extract from Parameter classes ('prior' or 'transform')
            default_factory: Optional factory for default values

        Returns:
            Instance of the generated attribute dataclass.
        """
        attr_cls = getattr(self.__class__, class_attr)
        hints = get_type_hints(self.__class__)
        values = {}

        # Build TypeVar -> actual type mapping
        typevar_map: dict[TypeVar, type] = {}
        orig_class = getattr(self, '__orig_class__', None)
        if orig_class is not None:
            type_params = getattr(self.__class__, '__parameters__', ())
            type_args = get_args(orig_class)
            typevar_map = dict(zip(type_params, type_args))

        for field in dataclass_fields(attr_cls):
            name = field.name
            ann = hints.get(name)
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                value = getattr(origin, attr_name, None)
                values[name] = (
                    value
                    if value is not None
                    else (default_factory() if default_factory else None)
                )

            elif isinstance(origin, TypeVar):
                actual_type = typevar_map.get(origin)
                if actual_type is not None and isinstance(actual_type, type):
                    if issubclass(actual_type, Parameter):
                        value = getattr(actual_type, attr_name, None)
                        values[name] = (
                            value
                            if value is not None
                            else (
                                default_factory() if default_factory else None
                            )
                        )

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                values[name] = nested_spec._build_attribute_instance(
                    class_attr, attr_name, default_factory
                )

        return attr_cls(**values)

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
