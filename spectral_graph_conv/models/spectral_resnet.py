import dataclasses
from typing import Callable

import torch.nn

from spectral_graph_conv.models.spectral_resnet_block import (
    SpectralResnetBlock,
    SpectralResnetBlockConfig,
)


@dataclasses.dataclass
class SpectralResnetConfig:
    n_layers: int
    filter_approximation_rank: int
    dtype: torch.dtype
    device: str
    init_std: float
    n_embed: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout: float
    ln_eps: float
    n_head: int


class SpectralResnet(torch.nn.Module):
    def __init__(self, config: SpectralResnetConfig):
        super().__init__()
        self.config = config
        self.block_config = SpectralResnetBlockConfig(
            filter_approximation_rank=config.filter_approximation_rank,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            n_embed=config.n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dropout=config.dropout,
            ln_eps=config.ln_eps,
            n_head=config.n_head,
        )
        self.blocks = torch.nn.ModuleList(
            [SpectralResnetBlock(self.block_config) for _ in range(config.n_layers)]
        )

    def init_weights(self) -> None:
        for block in self.blocks:
            block.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        inv_eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x, eigenvalues, eigenvectors, inv_eigenvectors)
        return x
