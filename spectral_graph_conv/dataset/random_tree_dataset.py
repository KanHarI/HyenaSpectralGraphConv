import dataclasses

import torch
import torch.utils.data

from spectral_graph_conv.dataset.graph_spectra import (
    adjacency_matrix_to_laplacian_spectra,
)
from spectral_graph_conv.dataset.toy_undirected_tree import ToyUndirectedTree


@dataclasses.dataclass
class RandomTreeDatasetConfig:
    n_nodes: int
    vocab_size: int
    dtype: torch.dtype


class RandomTreeSpectralDataset(
    torch.utils.data.Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    def __init__(self, config: RandomTreeDatasetConfig):
        super().__init__()
        self.config = config

    def __len__(self) -> int:
        return 1_000_000  # Hack for random datasets

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        random_tree = ToyUndirectedTree.create_random_tree(
            self.config.vocab_size, self.config.n_nodes
        )
        (
            eigenvalues,
            eigenvectors,
            inv_eigenvectors,
        ) = adjacency_matrix_to_laplacian_spectra(
            random_tree.edges, True, self.config.dtype, "cpu"
        )
        return (
            random_tree.nodes,
            random_tree.get_parent_node_labels_tensor(self.config.vocab_size),
            random_tree.edges,
            eigenvalues,
            eigenvectors,
            inv_eigenvectors,
        )
