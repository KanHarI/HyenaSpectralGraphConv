import dataclasses
from typing import Callable

import torch

from spectral_graph_conv.models.chebyshev_spectral_graph_conv import (
    ChebyshevSpectralGraphConv,
    ChebyshevSpectralGraphConvConfig,
)
from spectral_graph_conv.models.mlp import MLP, MLPConfig


@dataclasses.dataclass
class SpectralResnetBlockConfig:
    filter_approximation_rank: int
    dtype: torch.dtype
    device: str
    init_std: float
    n_embed: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout: float
    ln_eps: float


class SpectralResnetBlock(torch.nn.Module):
    def __init__(self, config: SpectralResnetBlockConfig):
        super().__init__()
        self.config = config
        self.ln = torch.nn.LayerNorm(
            config.n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.chebyshev_specgral_graph_conv_config = ChebyshevSpectralGraphConvConfig(
            filter_approximation_rank=config.filter_approximation_rank,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            n_embed=config.n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.chebyshev_specgral_graph_conv = ChebyshevSpectralGraphConv(
            self.chebyshev_specgral_graph_conv_config
        )
        self.node_space_mlp_config = MLPConfig(
            n_in=config.n_embed,
            n_out=config.n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            dropout=config.dropout,
            init_std=config.init_std,
        )
        self.node_space_mlp = MLP(self.node_space_mlp_config)

    def init_weights(self) -> None:
        self.chebyshev_specgral_graph_conv.init_weights()
        self.node_space_mlp.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        inv_eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        x = self.chebyshev_specgral_graph_conv(
            self.ln(x), eigenvalues, eigenvectors, inv_eigenvectors
        )
        x = self.node_space_mlp(x)
        return x
