from __future__ import annotations

from typing import Any, TypeVar

from ._prior import Prior

# Bounded TypeVars for learnable prior hyperparameters.
# These are bounded to Prior (not Parameter) because learnable
# hyperparameters are nested Prior Specs in the new architecture.

# For location parameters (mu, loc, etc.)
MuPrior = TypeVar("MuPrior", bound=Prior[Any])
LocPrior = TypeVar("LocPrior", bound=Prior[Any])

# For scale parameters (sigma, scale, etc.)
SigmaPrior = TypeVar("SigmaPrior", bound=Prior[Any])
ScalePrior = TypeVar("ScalePrior", bound=Prior[Any])
