from xabo.core.mean import Mean, ZeroMean

means = [
    ZeroMean(),
]


def mean_id(mean: Mean) -> str:
    return type(mean).__name__
