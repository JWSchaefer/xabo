from typing import TypeAlias

from jaxtyping import Array, Float

from xabo.core.gp._exact import ExactGP
from xabo.core.kernel import Kernel
from xabo.core.parameters import Parameter
from xabo.core.priors import Prior

TrainingFeatures: TypeAlias = Float[Array, '*S N X']
LengthscaleShape: TypeAlias = Float[Array, 'N']


class MyKernel(Kernel, Trainable):
    ell: Parameter[LengthscaleShape] | Prior[LengthscaleShape]
    ...


class MyGP(ExactGP, Trainable):
    kernel: Kernel
    ...


def main():
    gp = MyGP(kernel=MyKernel(ell=MyPrior(loc=[...], scale=[...])))
