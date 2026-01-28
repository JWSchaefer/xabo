from typing import Generic, Optional

from xabo.core.prior import Prior

from .._types import T
from ..transform._transform import Transform


class Parameter(Generic[T]):
    """Type marker for model parameters.

    Use in type hints to declare parameters:
        class MyModel(Spec):
            lengthscale: Parameter[float]

    Class attributes define defaults for transform and prior.
    """

    transform: Optional[Transform[T]] = None
    prior: Optional[Prior[T]] = None
