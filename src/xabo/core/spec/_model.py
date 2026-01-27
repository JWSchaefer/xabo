from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from xabo.core._types import Scalar

from ._transform import Transform

if TYPE_CHECKING:
    from ._spec import Spec


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Model:
    """Immutable container binding a Spec with params and state.

    Model provides:
    - Immutable parameter/state management via replace_*() methods
    - Transform operations (to/from unconstrained space)
    - Prior evaluation with Jacobian correction
    - Delegation to spec's __call__ method

    Usage:
        spec = MySpec(kernel=Matern12(rho=0.5, sigma=1.0), noise=0.01)
        model = Model.from_spec(spec)

        # Transform for optimization
        raw = model.to_unconstrained()

        # Update from optimizer
        model = model.from_unconstrained(new_raw_params)

        # Evaluate prior (includes Jacobian correction)
        lp = model.log_prior(raw_params)

        # Call the underlying spec
        result = model(x, y)
    """

    spec: Spec
    params: dict
    state: dict

    # --- Construction ---

    @classmethod
    def from_spec(cls, spec: Spec) -> Model:
        """Create Model from an instantiated Spec."""
        return cls(
            spec=spec,
            params=spec.init_params(),
            state=spec.init_state(),
        )

    # --- Immutable Updates ---

    def replace_params(self, params: dict) -> Model:
        """Return new Model with updated params."""
        return replace(self, params=params)

    def replace_state(self, state: dict) -> Model:
        """Return new Model with updated state."""
        return replace(self, state=state)

    # --- Transforms ---

    def to_unconstrained(self) -> dict:
        """Transform current params to unconstrained space."""
        transforms = self.spec._get_transforms()
        return self._apply_transforms(self.params, transforms, inverse=True)

    def from_unconstrained(self, raw_params: dict) -> Model:
        """Return new Model with params transformed from unconstrained space."""
        transforms = self.spec._get_transforms()
        constrained = self._apply_transforms(raw_params, transforms, inverse=False)
        return self.replace_params(constrained)

    def _apply_transforms(
        self, params: dict, transforms: dict, inverse: bool
    ) -> dict:
        """Apply transforms to params pytree."""
        result = {}
        for name, value in params.items():
            transform = transforms.get(name)

            if transform is None:
                result[name] = value
            elif isinstance(transform, dict):
                # Nested spec - recurse
                result[name] = self._apply_transforms(value, transform, inverse)
            elif isinstance(transform, Transform):
                if inverse:
                    result[name] = transform.inverse(value)
                else:
                    result[name] = transform.forward(value)
            else:
                result[name] = value

        return result

    # --- Prior Evaluation ---

    def log_prior(self, unconstrained_params: dict) -> Scalar:
        """Evaluate log prior with Jacobian correction.

        For MCMC/optimization in unconstrained space:
        log p(phi) = log p(f(phi)) + log|det J|

        where f transforms unconstrained -> constrained.

        Args:
            unconstrained_params: Parameters in unconstrained space.

        Returns:
            Log prior density with Jacobian correction.
        """
        # Transform to constrained space for prior evaluation
        transforms = self.spec._get_transforms()
        constrained = self._apply_transforms(
            unconstrained_params, transforms, inverse=False
        )

        # Evaluate prior in constrained space
        priors = self.spec._get_priors()
        prior_lp = self._eval_priors(constrained, priors)

        # Add Jacobian correction
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
                # Nested - recurse
                total = total + self._eval_priors(value, prior)
            else:
                # Prior instance
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
        """Delegate to spec's __call__ with current params and state."""
        if not callable(self.spec):
            raise TypeError(f'{type(self.spec).__name__} is not callable')
        return self.spec(self.state, self.params, *args, **kwargs)

    # --- JAX Pytree Registration ---

    def tree_flatten(self):
        """Flatten for JAX transformations.

        params and state are dynamic (can be traced by JAX).
        spec is static (defines structure, not traced).
        """
        return (self.params, self.state), self.spec

    @classmethod
    def tree_unflatten(cls, spec, children):
        """Unflatten from JAX transformations."""
        params, state = children
        return cls(spec=spec, params=params, state=state)
