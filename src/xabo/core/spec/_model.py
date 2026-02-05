from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Generic, cast

import jax
import jax.numpy as jnp

from xabo.core._types import Scalar

from ._params_structure import ParamsStructure
from ._spec import P, S, Spec, Tr


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Model(Generic[P, S, Tr]):
    """
    Immutable container binding a Spec with typed params and state.
    """

    spec: Spec[P, S, Tr]
    params: P
    state: S

    @classmethod
    def from_spec(cls, spec: Spec[P, S, Tr]) -> Model[P, S, Tr]:
        """Create Model from an instantiated Spec."""
        return cls(
            spec=spec,
            params=spec.init_params(),
            state=spec.init_state(),
        )

    def replace_params(self, params: P) -> Model[P, S, Tr]:
        """Return new Model with updated params."""
        return replace(self, params=params)

    def replace_state(self, state: S) -> Model[P, S, Tr]:
        """Return new Model with updated state."""
        return replace(self, state=state)

    def flatten_params(self) -> tuple[jnp.ndarray, ParamsStructure]:
        """Flatten params to a 1D array for use with external optimizers.

        Returns:
            flat: 1D array containing all parameter values.
            structure: Structure information needed for unflattening.

        """
        leaves, treedef = jax.tree_util.tree_flatten(self.params)

        arrays = []
        shapes = []
        dtypes = []

        for leaf in leaves:
            arr = jnp.atleast_1d(jnp.asarray(leaf))
            arrays.append(arr.ravel())
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)

        flat = jnp.concatenate(arrays) if arrays else jnp.array([])

        structure = ParamsStructure(
            treedef=treedef,
            shapes=tuple(shapes),
            dtypes=tuple(dtypes),
        )

        return flat, structure

    def unflatten_params(
        self, flat: jnp.ndarray, structure: ParamsStructure
    ) -> Model[P, S, Tr]:
        """Reconstruct Model with params from a 1D array.

        Args:
            flat: 1D array of parameter values.
            structure: Structure information from flatten_params.

        Returns:
            New Model with reconstructed params.
        """
        leaves = []
        offset = 0

        for shape, dtype in zip(structure.shapes, structure.dtypes):
            size = structure._prod(shape)
            leaf_flat = flat[offset : offset + size]
            leaf = leaf_flat.reshape(shape).astype(dtype)

            if shape == (1,):
                leaf = leaf[0]

            leaves.append(leaf)
            offset += size

        params = jax.tree_util.tree_unflatten(structure.treedef, leaves)
        return self.replace_params(params)

    def to_unconstrained(self) -> dict:
        """
        Transform current params to unconstrained space.
        Returns dict (not dataclass) for flexibility in optimization.
        """
        transforms = self.spec.get_transforms()
        return self._apply_transforms(self.params, transforms, inverse=True)

    def from_unconstrained(self, raw_params: dict) -> Model[P, S, Tr]:
        """Return new Model with params transformed from unconstrained space."""
        transforms = self.spec.get_transforms()
        constrained_dict = self._apply_transforms(
            raw_params, transforms, inverse=False
        )
        constrained = self.spec._dict_to_params(constrained_dict)
        return self.replace_params(constrained)

    def _apply_transforms(
        self, params: P | dict, transforms: Tr, inverse: bool
    ) -> dict:
        """Apply transforms to params.

        Args:
            params: Params dataclass or dict of param values
            transforms: Transforms dataclass instance
            inverse: If True, apply inverse transform (constrained -> unconstrained)

        Returns:
            Dict of transformed values (for optimization flexibility)
        """
        result = {}

        # Handle both dataclass and dict input for params
        if is_dataclass(params):
            items = [(f.name, getattr(params, f.name)) for f in fields(params)]
        else:
            items = cast(dict, params).items()

        for name, value in items:
            transform = getattr(transforms, name)

            if is_dataclass(transform):
                # Nested spec - recurse
                result[name] = self._apply_transforms(
                    value, transform, inverse
                )
            elif inverse:
                result[name] = transform.inverse(value)
            else:
                result[name] = transform.forward(value)

        return result

    def log_prior(self, unconstrained_params: dict) -> Scalar:
        """
        Evaluate log prior with Jacobian correction.

        Traverses the Spec tree to find all Prior instances and
        evaluates their log_prob on constrained parameter values.
        """
        transforms = self.spec.get_transforms()
        constrained_dict = self._apply_transforms(
            unconstrained_params, transforms, inverse=False
        )
        constrained_params = self.spec._dict_to_params(constrained_dict)

        prior_lp = self._eval_priors(self.spec, constrained_params)

        jacobian = self._log_det_jacobian(unconstrained_params, transforms)

        return prior_lp + jacobian

    def _eval_priors(self, spec: 'Spec', params: Any) -> Scalar:
        """Recursively evaluate all priors in the Spec tree.

        Walks the Spec tree and typed params in parallel. When a field
        is a Prior (which is a Spec), evaluates its log_prob and then
        recurses into its own fields to evaluate nested priors (hyperpriors).

        Args:
            spec: Current Spec node in the tree
            params: Corresponding typed Params dataclass

        Returns:
            Sum of all log prior probabilities
        """
        from ..prior._prior import Prior

        total = jnp.zeros(())

        for f in fields(params):
            name = f.name
            nested_spec = getattr(spec, name, None)
            nested_params = getattr(params, name, None)

            if nested_params is None:
                continue

            if isinstance(nested_spec, Prior):
                # This is a Prior - evaluate its log_prob
                value = jnp.asarray(nested_params.value)
                state = nested_spec.init_state()
                total = total + nested_spec.log_prob(
                    value, nested_params, state
                )
                # Recurse into the Prior's fields for nested priors (hyperpriors)
                total = total + self._eval_priors(nested_spec, nested_params)

            elif isinstance(nested_spec, Spec):
                # Non-Prior Spec (e.g., Kernel) - just recurse
                total = total + self._eval_priors(nested_spec, nested_params)

        return total

    def _log_det_jacobian(
        self, unconstrained_params: dict, transforms: Tr
    ) -> Scalar:
        """Sum of log|det J| for all transforms.

        Args:
            unconstrained_params: Dict of unconstrained parameter values
            transforms: Transforms dataclass instance

        Returns:
            Sum of log determinant of Jacobians
        """
        total = jnp.zeros(())

        for f in fields(cast(Any, transforms)):
            name = f.name
            value = unconstrained_params.get(name)
            transform = getattr(transforms, name)

            if value is None:
                continue
            elif is_dataclass(transform):
                # Nested - recurse
                total = total + self._log_det_jacobian(value, transform)
            else:
                total = total + transform.log_det_jacobian(value)

        return total

    # --- Delegation ---

    def __call__(self, *args, **kwargs) -> Any:
        if not callable(self.spec):
            raise TypeError(f'{type(self.spec).__name__} is not callable')
        return self.spec(self.state, self.params, *args, **kwargs)

    def tree_flatten(self):
        return self.params, self.state

    @classmethod
    def tree_unflatten(cls, spec, children):
        params, state = children
        return cls(spec=spec, params=params, state=state)
