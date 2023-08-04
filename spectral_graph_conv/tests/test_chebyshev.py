import numpy as np
import numpy.polynomial.chebyshev as cheb
import torch

from spectral_graph_conv.utils.chebyshev import (
    evaluate_chebyshev,
    naive_evaluate_chebyshev,
)


def test_naive_implementation_matches_numpy() -> None:
    for _ in range(10):
        # Generate random coefficients and a random x value
        coeffs = np.random.rand(10)
        x = np.random.rand()

        # Evaluate the polynomial using our function
        our_result = naive_evaluate_chebyshev(list(coeffs), x)

        # Evaluate the polynomial using NumPy's function
        numpy_result = cheb.chebval(x, coeffs)  # type: ignore

        # Check if the results are close
        assert np.isclose(
            our_result, numpy_result
        ), f"Results not close for coeffs={coeffs}, x={x}: our_result={our_result}, numpy_result={numpy_result}"


NUM_TESTS = 50
BATCH_SIZE = 5
NUM_COEFFS = 10
SAMPLED_POINTS = 4


def test_torch_implementation_chebyshev() -> None:
    for _ in range(NUM_TESTS):
        # Generate random coefficients and a random x value
        coeffs = np.random.rand(BATCH_SIZE, NUM_COEFFS)
        x = np.random.rand(BATCH_SIZE, SAMPLED_POINTS)

        # Evaluate the polynomial using our function
        our_result = evaluate_chebyshev(
            torch.tensor(coeffs).unsqueeze(1), torch.tensor(x)
        )

        # Evaluate the polynomial using NumPy's function
        numpy_result = np.array(
            [cheb.chebval(x[i], coeffs[i]) for i in range(BATCH_SIZE)]  # type: ignore
        )
        assert np.allclose(
            our_result.squeeze(1).squeeze(1).detach().numpy(), numpy_result
        ), f"Results not close for coeffs={coeffs}, x={x}: our_result={our_result}, numpy_result={numpy_result}"
