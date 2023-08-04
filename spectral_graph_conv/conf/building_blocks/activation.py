import dataclasses
from typing import Callable

import torch

from spectral_graph_conv.utils.new_expm1 import new_expm1
from spectral_graph_conv.utils.new_gelu import new_gelu
from spectral_graph_conv.utils.softplus import softplus


@dataclasses.dataclass
class ActivationConf:
    _activation: str

    @property
    def activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        match self._activation:
            case "relu":
                return torch.relu
            case "gelu":
                return torch.nn.functional.gelu
            case "elu":
                return torch.nn.functional.elu
            case "softplus":
                return softplus
            case "new_gelu":
                return new_gelu
            case "new_expm1":
                return new_expm1
            case _:
                raise ValueError(f"Unknown activation: {self._activation}")
