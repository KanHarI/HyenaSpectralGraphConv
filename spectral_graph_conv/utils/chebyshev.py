import torch


def naive_evaluate_chebyshev(coefficients: list[float], x: float) -> float:
    # Initialize a list to store the Chebyshev polynomials
    chebyshev_polynomials = [1, x]

    # Generate the required Chebyshev polynomials
    for i in range(2, len(coefficients)):
        next_polynomial = 2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2]
        chebyshev_polynomials.append(next_polynomial)

    # Sum the polynomials, weighted by the coefficients
    result = sum(
        coeff * polynomial
        for coeff, polynomial in zip(coefficients, chebyshev_polynomials)
    )

    return result


def evaluate_chebyshev(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    B, C, L = coefficients.shape
    _, N = x.shape  # (B, N)

    # Initialize a tensor to store the Chebyshev polynomials
    chebyshev_polynomials = torch.zeros(
        B, C, L, N, dtype=coefficients.dtype, device=coefficients.device
    )
    chebyshev_polynomials[..., 0, :] = 1
    chebyshev_polynomials[..., 1, :] = x.unsqueeze(-2)

    # Generate the required Chebyshev polynomials
    for i in range(2, L):
        chebyshev_polynomials[..., i, :] = (
            2 * x.unsqueeze(-2) * chebyshev_polynomials[..., i - 1, :]
            - chebyshev_polynomials[..., i - 2, :]
        )

    # Sum the polynomials, weighted by the coefficients
    result = torch.sum(coefficients.unsqueeze(-1) * chebyshev_polynomials, dim=-2)

    return result
