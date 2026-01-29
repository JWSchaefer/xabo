from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, cast

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
        return self._apply_transforms(self.params, transforms, inverse=True)

    def from_unconstrained(self, raw_params: dict) -> Model[P, S, Pr, Tr]:
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
                result[name] = self._apply_transforms(value, transform, inverse)
            elif inverse:
                result[name] = transform.inverse(value)
            else:
                result[name] = transform.forward(value)

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

    def _eval_priors(self, constrained_params: dict, priors: Pr) -> Scalar:
        """Evaluate priors in constrained space.

        Args:
            constrained_params: Dict of constrained parameter values
            priors: Priors dataclass instance

        Returns:
            Sum of log prior probabilities
        """
        total = jnp.zeros(())

        for f in fields(cast(Any, priors)):
            name = f.name
            value = constrained_params.get(name)
            prior = getattr(priors, name)

            if prior is None or value is None:
                continue
            elif is_dataclass(prior):
                # Nested priors - recurse
                total = total + self._eval_priors(value, prior)
            else:
                total = total + prior.log_prob(value)

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
