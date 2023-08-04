import torch


def new_expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) - 1.0
