import dataclasses

import torch
from typing_extensions import Self


@dataclasses.dataclass
class ToyUndirectedGraph:
    nodes: torch.Tensor  # shape: (N,) A tensor of labels for each node
    edges: torch.Tensor  # Shape: (N,N), adjacency matrix

    @classmethod
    def create_from_nodes_edges(
        cls, nodes: torch.Tensor, edges: list[tuple[int, int]]
    ) -> Self:
        tensor_edges = torch.zeros((len(nodes), len(nodes)), dtype=torch.bool)
        for i, j in edges:
            tensor_edges[i, j] = True
            tensor_edges[j, i] = True
        return cls(nodes=nodes, edges=tensor_edges)

    @classmethod
    def create_random_graph(cls, vocab_size: int, n_nodes: int, n_edges: int) -> Self:
        nodes = torch.randint(vocab_size, (n_nodes,), dtype=torch.int64)
        edges: list[tuple[int, int]] = []
        for _ in range(n_edges):
            i, j = torch.randint(n_nodes - 2, (2,))
            if j >= i:
                j += 1
            edges.append((i, j))
        return cls.create_from_nodes_edges(nodes, edges)
