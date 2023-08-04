import dataclasses
from typing import Callable

import torch

from spectral_graph_conv.models.spectral_resnet import (
    SpectralResnet,
    SpectralResnetConfig,
)
from spectral_graph_conv.models.toy_graph_embedder import (
    ToyGraphEmbedder,
    ToyGraphEmbedderConfig,
)


@dataclasses.dataclass
class ToyGraphSpectralResnetConfig:
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
    vocab_size: int
    nll_epsilon: float
    embedder_sigma: float
    max_depth: int
    n_head: int


class ToyGraphSpectralResnet(torch.nn.Module):
    def __init__(self, config: ToyGraphSpectralResnetConfig):
        super().__init__()
        self.config = config
        self.toy_graph_embedder_config = ToyGraphEmbedderConfig(
            vocab_size=config.vocab_size,
            n_embed=config.n_embed,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            nll_epsilon=config.nll_epsilon,
            embedder_sigma=config.embedder_sigma,
            max_depth=config.max_depth,
        )
        self.toy_graph_embedder = ToyGraphEmbedder(self.toy_graph_embedder_config)
        self.spectral_resnet_config = SpectralResnetConfig(
            n_layers=config.n_layers,
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
        self.spectral_resnet = SpectralResnet(self.spectral_resnet_config)

    def init_weights(self) -> None:
        self.toy_graph_embedder.init_weights()
        self.spectral_resnet.init_weights()

    def forward(
        self,
        input_nodes: torch.Tensor,
        input_depths: torch.Tensor,
        output_nodes: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        inv_eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        x = self.toy_graph_embedder(input_nodes, input_depths)
        x = self.spectral_resnet(x, eigenvalues, eigenvectors, inv_eigenvectors)
        return self.toy_graph_embedder.loss(x, output_nodes)
