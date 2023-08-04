import dataclasses
import typing
from typing import Any

import torch


@dataclasses.dataclass
class OptimizerConf:
    _optimizer: str
    lr: float
    warmup_iters: int
    max_iters: int
    schedule: str
    weight_decay: float
    eps: float
    beta1: float
    beta2: float
    batch_size: int
    grad_accumulation_steps: int

    def create_optimizer(self, parameters: Any) -> torch.optim.Optimizer:
        match self._optimizer:
            case "adamw":
                return torch.optim.AdamW(
                    parameters,
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case "adam":
                return torch.optim.Adam(
                    parameters,
                    lr=self.lr,
                    betas=(self.beta1, self.beta2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case "sgd":
                return torch.optim.SGD(
                    parameters,
                    lr=self.lr,
                    momentum=self.beta1,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise ValueError(f"Unknown optimizer: {self._optimizer}")

    def get_lr(self, step: int) -> float:
        match self.schedule:
            case "linear":
                if step < self.warmup_iters:
                    return self.lr * (step / self.warmup_iters)
                else:
                    return self.lr * (
                        1
                        - (step - self.warmup_iters)
                        / (self.max_iters - self.warmup_iters)
                    )
            case "constant":
                if step < self.warmup_iters:
                    return self.lr * (step / self.warmup_iters)
                else:
                    return self.lr
            case "invsqrt":
                if step < self.warmup_iters:
                    return self.lr * (step / self.warmup_iters)
                else:
                    return typing.cast(
                        float, self.lr / ((step / self.warmup_iters) ** 0.5)
                    )
            case _:
                raise ValueError(f"Unknown schedule: {self.schedule}")
