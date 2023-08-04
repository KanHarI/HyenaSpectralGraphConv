import dataclasses

import torch


@dataclasses.dataclass
class ToyGraphEmbedderConfig:
    vocab_size: int
    n_embed: int
    dtype: torch.dtype
    device: str
    init_std: float
    nll_epsilon: float
    embedder_sigma: float
    max_depth: int


class ToyGraphEmbedder(torch.nn.Module):
    def __init__(self, config: ToyGraphEmbedderConfig):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Parameter(
            torch.zeros(
                (config.vocab_size, config.n_embed),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.depth_embeddings = torch.nn.Parameter(
            torch.zeros(
                (config.max_depth, config.n_embed),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.noise_projection = torch.nn.Parameter(
            torch.zeros(
                (config.n_embed, config.n_embed),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            ),
        )
        self.nll_epsilon = config.nll_epsilon
        self.embedder_sigma = config.embedder_sigma

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.embeddings, std=self.config.init_std)
        torch.nn.init.normal_(self.noise_projection, std=self.config.init_std)

    def forward(self, nodes: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        result = self.embeddings[nodes] + self.depth_embeddings[depths]
        noise = torch.randn_like(result)
        return (
            result
            + torch.einsum("bni,ij->bnj", noise, self.noise_projection)
            * self.embedder_sigma
        )

    def loss(self, x: torch.Tensor, y_discrete: torch.Tensor) -> torch.Tensor:
        restored = torch.einsum("bnv,lv->bnl", x, self.embeddings)
        restored_softmax = torch.softmax(restored.real, dim=-1)  # (B, N, L)
        # restored_softmax: shape (B, N, L)
        # y_discrete: shape (B, N)
        gathered = torch.gather(
            restored_softmax, dim=-1, index=y_discrete.unsqueeze(-1)
        ).squeeze(-1)
        nll = -torch.log(gathered + self.nll_epsilon)
        return nll.mean()
