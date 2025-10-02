import torch
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utility import ulp_delta, worst_ulp_delta


def build_polynomial_approx_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype="float32"):

    np_dtype = getattr(np, dtype)

    npoints = 1000
    np_inputs = np.linspace(xmin, xmax, npoints)
    np_outputs = np.vectorize(fun)(np_inputs)

    print(f"===== fun: {fun}")

    if dtype == "float32":
        eps = 1.1920928955078125e-07
    elif dtype == "float64":
        eps = 2e-16
    elif dtype == "bfloat16":
        eps = 0.0078125
    else:
        eps = None

    if eps is not None:
        rcond = eps * len(np_inputs)

    chebyshev_coeffs = np.polynomial.chebyshev.chebfit(np_inputs, np_outputs, poly_rank, rcond=rcond)
    coeffs = np.flip(np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs))

    test_inputs = np.linspace(xmin, xmax, 10)
    cheby_result = np.polynomial.chebyshev.chebval(test_inputs, chebyshev_coeffs)
    result = np.polyval(coeffs, test_inputs)
    print(f"  coeffs: {coeffs}")
    print(f"  test_inputs: {test_inputs}")
    print(f"  np.polyval: {result}")
    print(f"  np.polynomial.chebyshev.chebval: {cheby_result}")

    return coeffs.astype(np_dtype).tolist()


def build_polynomial_approx(fun, poly_rank, xmin, xmax, npoints, dtype="float32", minimize_method='BFGS'):
    """
    Build a polynomial approximation of a function using least squares optimization.

    Args:
        fun: The function to approximate
        poly_rank: Number of polynomial coefficients
        xmin: Minimum input value of the range
        xmax: Maximum input value of the range
        npoints: Number of data points to use in the range

    Returns:
        List of polynomial coefficients [c0, c1, c2, ...] such that
        sum(coeff * x^i for i, coeff in enumerate(coeffs)) approximates fun(x)
    """
    # Generate uniformly spaced points in the range
    np_dtype = getattr(np, dtype)
    torch_dtype = getattr(torch, dtype)

    np_x_points = np.linspace(xmin, xmax, npoints)
    np_y_golden_points = np.vectorize(fun)(np_x_points)

    def objective(coeffs):
        """Objective function: maximum relative error"""

        np_polyval = np.polyval(coeffs.astype(np_dtype), np_x_points.astype(np_dtype))
        torch_calculated = torch.tensor(np_polyval, dtype=torch_dtype)
        relative_errors = abs(torch_calculated - torch.tensor(np_y_golden_points, dtype=torch_dtype))

        return torch.max(relative_errors)

    def objective_ulp(coeffs):
        """Objective function: maximum ULP error"""

        np_polyval = np.polyval(coeffs.astype(np_dtype), np_x_points.astype(np_dtype))
        torch_calculated = torch.tensor(np_polyval, dtype=torch_dtype)
        ulp_errors = ulp_delta(torch_calculated, torch.tensor(np_y_golden_points, dtype=torch_dtype))

        # Also try to minimize ULP error at key points (e.g. 0)
        # To do this, we manually compute the value, and weights the ULP error by a factor 4
        critical_points = [0]
        critical_factor = 1
        critical_values = np.polyval(coeffs.astype(np_dtype), np.array([0], dtype=np_dtype))
        critical_ulp_errors = ulp_delta(torch.tensor(critical_values, dtype=torch_dtype), torch.tensor(np_y_golden_points, dtype=torch_dtype)) * critical_factor

        return torch.max(ulp_errors)

    # Initial guess: start with zeros
    initial_guess = np.zeros(poly_rank)

    # Use least squares as initial approximation for better starting point
    # Build Vandermonde matrix for least squares
    A = np.vander(np_x_points, poly_rank, increasing=True)
    initial_guess = np.linalg.lstsq(A, np_y_golden_points, rcond=None)[0]

    # Minimize the maximum relative error
    result = minimize_scalar(objective_ulp, initial_guess, method=minimize_method)

    return result.x.tolist()


def compare_ulp_error(fun, coeffs, xmin, xmax, npoints, dtype=torch.float32):
    """
    Compare ULP error of polynomial approximation against the golden function.

    Args:
        fun: The golden function
        coeffs: Polynomial coefficients from build_polynomial_approx
        xmin: Minimum input value of the range
        xmax: Maximum input value of the range
        npoints: Number of test points

    Returns:
        Maximum ULP delta across the test points
    """
    # Generate test points
    x_points = np.linspace(xmin, xmax, npoints)

    # Compute golden values
    golden_values = torch.tensor([fun(x) for x in x_points], dtype=torch.float64)

    # Compute polynomial approximation values
    approx_values = torch.tensor(np.polyval(coeffs, x_points), dtype=dtype).to(dtype)

    # Compute worst ULP delta
    return worst_ulp_delta(approx_values, golden_values)


def generate_polynomial_approx(fun, poly_rank, xrange, npoints, dtype="float32", minimize_method='BFGS'):

    xmin, xmax = xrange

    # Test with exponential function
    coeffs = build_polynomial_approx_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype="float32")
    print(f"rank {poly_rank} - Polynomial coefficients: {coeffs}")

    # Test ULP error
    ulp_error = compare_ulp_error(fun, coeffs, xmin, xmax, npoints)
    print(f"    Maximum ULP error: {ulp_error}")

    return coeffs
