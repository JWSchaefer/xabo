import torch


def _f(x: torch.tensor):
    return torch.pow((6 * x - 2), 2) * torch.sin(12 * x - 4)


def forrester(x: torch.tensor, a: float = 1.0, b: float = 0.0, c: float = 0.0):
    return a * _f(x) + b * (x - 0.5) - c
