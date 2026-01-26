from typing import Type

from jax.tree_util import tree_leaves
from jaxtyping import Array, Float

from xabo.core.kernel import Matern12
from xabo.core.spec._parameter import Parameter
from xabo.core.spec._spec import Spec


class Scalar(Parameter[float]):
    @classmethod
    def default(cls: Type['Scalar'], rng=None) -> float:
        return 0.0


class MyModel(Spec):
    kernel: Matern12[Scalar, Scalar]
    other: Parameter[Float[Array, '2']]


if __name__ == '__main__':

    # state = MyModel.to_state()
    params = MyModel.to_param_structure()
    print(params)
    print(tree_leaves(params))
    # print(params, state)
