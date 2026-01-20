from typing import TypeAlias

from jaxtyping import Array, Float

TrainingFeatures: TypeAlias = Float[Array, '*S N X']
TrainingObservations: TypeAlias = Float[Array, '*S N 1']

TestFeatures: TypeAlias = Float[Array, '*S M X']
TestPredictions: TypeAlias = Float[Array, '*S M 1']


LowerDecomposition: TypeAlias = Float[Array, '*S N N']
