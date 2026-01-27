import jax.numpy as jnp
from jaxtyping import Array, Float

from xabo.core import spec
from xabo.core._types import Scalar
from xabo.core.kernel import Matern12
from xabo.core.prior._lognormal import LogNormal
from xabo.core.spec import Spec, State
from xabo.core.spec._model import Model
from xabo.core.spec._parameter import Parameter
from xabo.core.spec._transform import Log


class ObservationNoise(Parameter[Scalar]):
    transform = Log()
    prior = LogNormal(mu=0.0, sigma=2.0)


class MyModelSpec(Spec):
    kernel: Matern12[float, float]
    observation_noise: ObservationNoise

    x: State[Float[Array, '*S N X']]
    y: State[Float[Array, '*S N 1']]
    lower: State[Float[Array, '*S N N']]

    def __call__(
        self,
    ):
        pass


if __name__ == '__main__':
    spec = MyModelSpec(
        kernel=Matern12(rho=0.5, sigma=1.0),
        observation_noise=0.0,
    )

    model = Model.from_spec(spec)

    print('Params:')
    print(model.params)

    # Extract state pytree (initially None)
    state = spec.init_state()
    print('\nState:')
    print(model.state)
