from typing import Generic, Optional, TypeVar

from xabo.core.prior import Prior

from ..transform._transform import Transform

T = TypeVar('T')


class Parameter(Generic[T]):
    """Type marker for model parameters.

    Use in type hints to declare parameters:
        class MyModel(Spec):
            lengthscale: Parameter[float]

    Class attributes define defaults for transform and prior.
    """

    transform: Optional[Transform[T]] = None
    prior: Optional[Prior[T]] = None
