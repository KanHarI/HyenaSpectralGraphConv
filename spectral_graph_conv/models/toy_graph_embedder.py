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
        self.nll_epsilon = config.nll_epsilon

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.embeddings, std=self.config.init_std)

    def forward(self, discrete: torch.Tensor) -> torch.Tensor:
        return self.embeddings[discrete]

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
