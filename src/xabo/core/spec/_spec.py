from abc import ABC, ABCMeta
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass
from typing import (
    Any,
    ClassVar,
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

from ..transform import Identity
from ..transform._transform import Transform
from ._parameter import Parameter
from ._state import State

P = TypeVar("P")
S = TypeVar("S")
Tr = TypeVar("Tr")

_params_class_cache: dict[type, type] = {}
_state_class_cache: dict[type, type] = {}
_transforms_class_cache: dict[type, type] = {}
_prior_class: list = []  # Cached Prior class to avoid repeated imports

_GENERATED_ATTRS = ("Params", "State", "Transforms")


def _get_prior_class():
    """Get the cached Prior class, importing lazily if needed."""
    if not _prior_class:
        try:
            from ..prior import Prior

            _prior_class.append(Prior)
        except ImportError:
            _prior_class.append(object)
    return _prior_class[0]


def _is_spec_bound(tv: TypeVar) -> bool:
    """Check if a TypeVar is bounded to Spec (including Prior)."""
    bound = getattr(tv, "__bound__", None)
    if bound is None:
        return False
    bound_origin = get_origin(bound) or bound
    return isinstance(bound_origin, type) and issubclass(bound_origin, Spec)


def _is_parameter_bound(tv: TypeVar) -> bool:
    """Check if a TypeVar is bounded to Parameter."""
    bound = getattr(tv, "__bound__", None)
    if bound is None:
        return False
    bound_origin = get_origin(bound) or bound
    return isinstance(bound_origin, type) and issubclass(bound_origin, Parameter)


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
            if _is_parameter_bound(origin) or _is_spec_bound(origin):
                # Both Parameter-bounded and Spec/Prior-bounded TypeVars
                # get Any type (resolved at runtime)
                fields.append((name, Any))

        elif isinstance(origin, type) and issubclass(origin, Spec):
            # Nested Spec (including Prior) - recursively generate its Params class
            nested_params = _generate_params_class(origin)
            fields.append((name, nested_params))

    class_name = f"{spec_cls.__name__}Params"
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

    class_name = f"{spec_cls.__name__}State"
    StateClass = make_dataclass(class_name, fields, frozen=True)

    # Register as JAX PyTree
    jax.tree_util.register_dataclass(
        StateClass,
        data_fields=[f[0] for f in fields],
        meta_fields=[],
    )

    _state_class_cache[spec_cls] = StateClass
    return StateClass


def _generate_transforms_class(spec_cls: type) -> type:
    """Generate frozen dataclass for a Spec's transforms.

    In the new architecture, transforms come from Prior class attributes,
    not from Parameter class attributes.
    """
    if spec_cls in _transforms_class_cache:
        return _transforms_class_cache[spec_cls]

    hints = get_type_hints(spec_cls)
    fields = []

    for name, ann in hints.items():
        if name in _GENERATED_ATTRS:
            continue

        origin = get_origin(ann) or ann

        if isinstance(origin, type) and issubclass(origin, Parameter):
            # Terminal Parameter - always has a transform (Identity by default)
            fields.append((name, Transform))

        elif isinstance(origin, TypeVar):
            if _is_parameter_bound(origin) or _is_spec_bound(origin):
                fields.append((name, Any))

        elif isinstance(origin, type) and issubclass(origin, Spec):
            nested_transforms = _generate_transforms_class(origin)
            if dataclass_fields(nested_transforms):
                fields.append((name, nested_transforms))
            else:
                # Even empty, include for consistency
                fields.append((name, nested_transforms))

    class_name = f"{spec_cls.__name__}Transforms"
    TransformsClass = make_dataclass(class_name, fields, frozen=True)

    _transforms_class_cache[spec_cls] = TransformsClass
    return TransformsClass


class SpecMeta(ABCMeta):
    """Metaclass that generates Params, State, and Transforms classes for Specs."""

    ATTRS = _GENERATED_ATTRS

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes (Spec, Prior)
        if name in ("Spec", "Prior"):
            return cls

        # Ensure Prior class is cached for later use
        _get_prior_class()

        # Generate and attach Params/State/Transforms classes
        setattr(cls, "Params", _generate_params_class(cls))
        setattr(cls, "State", _generate_state_class(cls))
        setattr(cls, "Transforms", _generate_transforms_class(cls))

        return cls


class Spec(ABC, Generic[P, S, Tr], metaclass=SpecMeta):
    """Base class for declarative model definitions.

    Subclasses define model structure via type hints:

    Fields can be:
    - Parameter[T]: Terminal leaf values (in Params tree)
    - State[T]: Cached/derived values (in State tree)
    - Nested Spec (including Prior): Branches with their own structure

    All learnable values are terminal Parameter[T] leaves.
    All branching/composite structures are Specs.

    Example:
        class MyModel(Spec):
            kernel: Matern52[LogNormal[float]]
            noise: LogNormal[float]

        model = MyModel(
            kernel=Matern52(
                lengthscale=LogNormal(value=0.5, mu=0.0, sigma=1.0),
                sigma=LogNormal(value=1.0, mu=0.0, sigma=1.0),
            ),
            noise=LogNormal(value=0.01, mu=-4.0, sigma=0.5),
        )

        params = model.init_params()
        params.kernel.lengthscale.value  # 0.5
        params.noise.value               # 0.01
    """

    Params: type[P]  # Set by metaclass
    State: type[S]  # Set by metaclass
    Transforms: type[Tr]  # Set by metaclass

    def __init__(self, **kwargs):
        """Initialize with values for Parameters and nested Specs.

        Args:
            **kwargs: Values for each Parameter and Spec field defined in type hints.

        Raises:
            TypeError: If required arguments are missing or unexpected arguments provided.
        """
        hints = get_type_hints(self.__class__)
        expected = {
            n
            for n, ann in hints.items()
            if n not in SpecMeta.ATTRS and get_origin(ann) is not ClassVar
        }

        # Check for unexpected arguments
        unexpected = set(kwargs) - expected
        if unexpected:
            raise TypeError(f"Unexpected argument(s): {', '.join(sorted(unexpected))}")

        for name, ann in hints.items():
            # Skip class attributes set by metaclass
            if name in SpecMeta.ATTRS:
                continue

            # Skip ClassVar annotations (class-level, not instance)
            if get_origin(ann) is ClassVar:
                continue

            origin = get_origin(ann) or ann

            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif isinstance(origin, type) and issubclass(origin, State):
                setattr(self, name, None)
            else:
                raise TypeError(f"Missing required argument: {name}")

    def init_params(self, rng: Optional[Array] = None) -> P:
        """Extract parameter values into a typed Params pytree.

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
            raise ValueError("rng required when from_prior=True")

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
                        "Sampling from priors not yet implemented"
                    )
                params[name] = value

            elif isinstance(origin, TypeVar):
                if _is_parameter_bound(origin):
                    # Parameter-bounded TypeVar - leaf value
                    value = getattr(self, name)
                    params[name] = value
                elif _is_spec_bound(origin):
                    # Spec/Prior-bounded TypeVar - recurse
                    nested_spec = getattr(self, name)
                    params[name] = nested_spec._collect_params_dict(rng, from_prior)

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                params[name] = nested_spec._collect_params_dict(rng, from_prior)

        return params

    def _dict_to_params(self, params_dict: dict) -> P:
        """Convert nested dict to typed Params dataclass."""
        hints = get_type_hints(self.__class__)
        converted = {}

        for name, value in params_dict.items():
            ann = hints.get(name)
            if ann is None:
                converted[name] = value
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                converted[name] = nested_spec._dict_to_params(value)
            elif isinstance(origin, TypeVar):
                if _is_spec_bound(origin):
                    nested_spec = getattr(self, name)
                    converted[name] = nested_spec._dict_to_params(value)
                else:
                    converted[name] = value
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

    def get_transforms(self) -> Tr:
        """Get transforms with named attribute access.

        Returns:
            Transforms dataclass with named attributes for each parameter.
        """
        return self._build_transforms()

    def _build_transforms(self) -> Any:
        """Build transforms instance for this Spec.

        For Parameter fields, uses Identity transform.
        For nested Spec/Prior fields, recurses.
        For Prior fields, the Prior's get_transforms() handles
        using the Prior's class-level transform for its value field.
        """
        transforms_cls = getattr(self.__class__, "Transforms")
        hints = get_type_hints(self.__class__)
        values = {}

        for field in dataclass_fields(transforms_cls):
            name = field.name
            ann = hints.get(name)
            if ann is None:
                values[name] = Identity()
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                # Terminal parameter - use Identity
                values[name] = Identity()

            elif isinstance(origin, TypeVar):
                # TypeVar - resolve at runtime
                nested = getattr(self, name, None)
                if isinstance(nested, Spec):
                    values[name] = nested._build_transforms()
                else:
                    values[name] = Identity()

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                values[name] = nested_spec._build_transforms()

            else:
                values[name] = Identity()

        return transforms_cls(**values)

    @classmethod
    def to_spec(cls: Type["Spec"]) -> dict:
        """Extract type specification from class annotations.

        Returns:
            Dict mapping field names to their type markers (Parameter/State/nested Spec).
        """
        return {
            name: cls._from_annotation(ann) for name, ann in get_type_hints(cls).items()
        }

    @classmethod
    def _from_annotation(cls, ann):
        origin = get_origin(ann) or ann
        args = get_args(ann)

        if isinstance(origin, type) and issubclass(origin, (Parameter, State)):
            return origin

        if isinstance(origin, type) and issubclass(origin, Spec):
            return origin.to_structure_from_args(args)

        raise TypeError(f"Unsupported annotation {ann}")

    @classmethod
    def to_structure_from_args(cls, args):
        type_vars = getattr(cls, "__parameters__", ())
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

        return {name: cls._from_annotation(ann) for name, ann in resolved_hints.items()}
