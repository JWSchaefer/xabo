from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Generic

import jax
import jax.numpy as jnp

from xabo.core._types import Scalar

from ..transform import Transform
from ._params_structure import ParamsStructure
from ._spec import P, Pr, S, Tr

if TYPE_CHECKING:
    from ._spec import Spec


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Model(Generic[P, S, Pr, Tr]):
    """
    Immutable container binding a Spec with typed params and state.
    """

    spec: Spec[P, S, Pr, Tr]
    params: P
    state: S

    @classmethod
    def from_spec(cls, spec: Spec[P, S, Pr, Tr]) -> Model[P, S, Pr, Tr]:
        """Create Model from an instantiated Spec."""
        return cls(
            spec=spec,
            params=spec.init_params(),
            state=spec.init_state(),
        )

    def replace_params(self, params: P) -> Model[P, S, Pr, Tr]:
        """Return new Model with updated params."""
        return replace(self, params=params)

    def replace_state(self, state: S) -> Model[P, S, Pr, Tr]:
        """Return new Model with updated state."""
        return replace(self, state=state)

    # --- Flatten/Unflatten for external optimizers ---

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
    ) -> Model[P, S, Pr, Tr]:
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

            # If original was scalar, extract it
            if shape == (1,):
                leaf = leaf[0]

            leaves.append(leaf)
            offset += size

        params = jax.tree_util.tree_unflatten(structure.treedef, leaves)
        return self.replace_params(params)

    # --- Transforms ---

    def to_unconstrained(self) -> dict:
        """
        Transform current params to unconstrained space.
        Returns dict (not dataclass) for flexibility in optimization.
        """
        transforms = self.spec.get_transforms()
        params_dict = self._dataclass_to_dict(self.params)
        return self._apply_transforms_dict(
            params_dict, transforms, inverse=True
        )

    def from_unconstrained(self, raw_params: dict) -> Model[P, S, Pr, Tr]:
        """Return new Model with params transformed from unconstrained space."""
        transforms = self.spec.get_transforms()
        constrained_dict = self._apply_transforms_dict(
            raw_params, transforms, inverse=False
        )
        constrained = self.spec._dict_to_params(constrained_dict)
        return self.replace_params(constrained)

    def _apply_transforms_dict(
        self, params: dict, transforms: dict, inverse: bool
    ) -> dict:
        """Apply transforms to params dict."""
        result = {}
        for name, value in params.items():
            transform = transforms.get(name)

            if transform is None:
                result[name] = value
            elif isinstance(transform, dict):
                # Nested spec - recurse
                result[name] = self._apply_transforms_dict(
                    value, transform, inverse
                )
            elif isinstance(transform, Transform):
                if inverse:
                    result[name] = transform.inverse(value)
                else:
                    result[name] = transform.forward(value)
            else:
                result[name] = value

        return result

    def _dataclass_to_dict(self, obj: Any) -> dict:
        """Recursively convert dataclass to nested dict."""
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if hasattr(value, '__dataclass_fields__'):
                result[f.name] = self._dataclass_to_dict(value)
            else:
                result[f.name] = value
        return result

    def log_prior(self, unconstrained_params: dict) -> Scalar:
        """
        Evaluate log prior with Jacobian correction.
        """
        transforms = self.spec.get_transforms()
        constrained = self._apply_transforms(
            unconstrained_params, transforms, inverse=False
        )

        priors = self.spec.get_priors()
        prior_lp = self._eval_priors(constrained, priors)

        jacobian = self._log_det_jacobian(unconstrained_params, transforms)

        return prior_lp + jacobian

    def _eval_priors(self, constrained_params: dict, priors: dict) -> Scalar:
        """Evaluate priors in constrained space."""
        total = jnp.zeros(())

        for name, value in constrained_params.items():
            prior = priors.get(name)

            if prior is None:
                continue
            elif isinstance(prior, dict):
                total = total + self._eval_priors(value, prior)
            else:
                total = total + prior.log_prob(value)

        return total

    def _log_det_jacobian(
        self, unconstrained_params: dict, transforms: dict
    ) -> Scalar:
        """Sum of log|det J| for all transforms."""
        total = jnp.zeros(())

        for name, value in unconstrained_params.items():
            transform = transforms.get(name)

            if transform is None:
                continue
            elif isinstance(transform, dict):
                # Nested - recurse
                total = total + self._log_det_jacobian(value, transform)
            elif isinstance(transform, Transform):
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
