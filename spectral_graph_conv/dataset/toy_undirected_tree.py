import dataclasses
import typing

import torch
from typing_extensions import Self

from spectral_graph_conv.dataset.toy_undirected_graph import ToyUndirectedGraph


@dataclasses.dataclass
class ToyTree(ToyUndirectedGraph):
    @classmethod
    def create_random_tree(cls, vocab_size: int, n_nodes: int) -> Self:
        nodes = torch.randint(vocab_size, (n_nodes,))
        edges: list[tuple[int, int]] = []
        for i in range(1, n_nodes):
            parent = typing.cast(int, torch.randint(i, (1,)).item())
            edges.append((parent, i))
        return cls.create_from_nodes_edges(nodes, edges)

    def get_parent_node_indices(self) -> torch.Tensor:
        # Return a tensor of shape (n_nodes,) where the i-th element is the index of the parent of the i-th node
        # We find the parent of every node by finding the first node with a smaller index that is connected to it
        parent_nodes = torch.zeros((len(self.nodes),), dtype=torch.int64)
        parent_nodes[0] = -1
        for i in range(1, len(self.nodes)):
            for j in range(i - 1, -1, -1):
                if self.edges[j, i]:
                    parent_nodes[i] = j
                    break
        return parent_nodes

    def get_parent_node_labels_tensor(self, vocab_size: int) -> torch.Tensor:
        root_key = vocab_size  # One past the vocab size
        # Return a tensor of shape (n_nodes,) where the i-th element is the label of the parent of the i-th node
        # We find the parent of every node by finding the first node with a smaller index that is connected to it
        parent_nodes = torch.zeros((len(self.nodes),), dtype=torch.int64)
        parent_nodes[0] = root_key
        for i in range(1, len(self.nodes)):
            for j in range(i - 1, -1, -1):
                if self.edges[j, i]:
                    parent_nodes[i] = self.nodes[j]
                    break
        return parent_nodes
