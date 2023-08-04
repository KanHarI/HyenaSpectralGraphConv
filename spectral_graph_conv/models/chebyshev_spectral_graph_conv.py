import dataclasses
from typing import Callable

import torch

from spectral_graph_conv.models.mlp import MLP, MLPConfig
from spectral_graph_conv.utils.chebyshev import evaluate_chebyshev


@dataclasses.dataclass
class ChebyshevSpectralGraphConvConfig:
    filter_approximation_rank: int
    dtype: torch.dtype
    device: str
    init_std: float
    n_embed: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout: float


class ChebyshevSpectralGraphConv(torch.nn.Module):
    def __init__(self, config: ChebyshevSpectralGraphConvConfig):
        super().__init__()
        self.config = config
        self.coefficients = torch.nn.Parameter(
            torch.zeros(
                (config.n_embed, config.filter_approximation_rank),
                dtype=torch.float32,
                device=config.device,
                requires_grad=True,
            )
        )
        self.fourier_space_mlp_config = MLPConfig(
            n_in=config.n_embed,
            n_out=config.n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            dropout=config.dropout,
            init_std=config.init_std,
        )
        self.fourier_space_mlp = MLP(self.fourier_space_mlp_config)

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.coefficients, std=self.config.init_std)
        self.fourier_space_mlp.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        inv_eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, C)
        # eigenvalues: (B, N)
        # eigenvectors: (B, N, N)
        # inv_eigenvectors: (B, N, N)
        x = x.view(x.shape[0], x.shape[1], self.config.n_embed)  # (B, N, C)
        x_fourier_space = torch.einsum("bni,bnk->bki", x, inv_eigenvectors)  # (B, N, C)
        stacked_coefficients = self.coefficients.unsqueeze(0).repeat(x.shape[0], 1, 1)
        amplification_per_ev = evaluate_chebyshev(
            stacked_coefficients, eigenvalues
        )  # (B, C, N)
        fourier_convolution = torch.einsum(
            "bin,bni->bni", amplification_per_ev, x_fourier_space
        )
        fourier_convolution = fourier_convolution + self.fourier_space_mlp(
            fourier_convolution
        )
        x_restored = torch.einsum(
            "bni,bnk->bki", fourier_convolution, eigenvectors
        ).contiguous()
        return x_restored
