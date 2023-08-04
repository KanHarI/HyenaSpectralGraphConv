import torch


def adjacency_matrix_to_laplacian(
    adj_matrix: torch.Tensor, is_symmetric: bool, dtype: torch.dtype
) -> torch.Tensor:
    adj_matrix_with_dtype = adj_matrix.to(dtype=dtype)
    if is_symmetric:
        adj_matrix_with_dtype = (
            adj_matrix_with_dtype + adj_matrix_with_dtype.transpose(-1, -2)
        ) / 2
    degree = adj_matrix.sum(dim=-1)
    inv_sqrt_degree = degree.pow(-0.5)
    inv_sqrt_degree[~torch.isfinite(inv_sqrt_degree)] = 0
    normalized_adj_matrix = adj_matrix_with_dtype * inv_sqrt_degree.unsqueeze(-1)
    normalized_adj_matrix = normalized_adj_matrix * inv_sqrt_degree.unsqueeze(-2)
    return (
        torch.eye(
            normalized_adj_matrix.shape[-1], dtype=dtype, device=adj_matrix.device
        )
        - normalized_adj_matrix
    )


def adjacency_matrix_to_laplacian_spectra(
    adj_batch: torch.Tensor, is_symmetric: bool, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    laplacian_batch = adjacency_matrix_to_laplacian(adj_batch, is_symmetric, dtype)
    if is_symmetric:
        values, vectors = torch.linalg.eigh(laplacian_batch)
        inv_vectors = vectors.transpose(-1, -2)  # Exploiting symmetry
        return values, vectors, inv_vectors
    else:
        values, vectors = torch.linalg.eig(laplacian_batch)
        inv_vectors = torch.linalg.inv(vectors)
        return values, vectors, inv_vectors
