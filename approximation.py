import torch
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utility import ulp_delta, worst_ulp_delta, compare_ulp_error



def generate_poly_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype="float32"):
    np_dtype = getattr(np, dtype)

    npoints = 1000
    np_inputs = np.linspace(xmin, xmax, npoints)
    np_outputs = np.vectorize(fun)(np_inputs)

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

    return coeffs.astype(np_dtype).tolist()


def generate_polynomial_approx(fun, poly_rank, xmin, xmax, npoints, dtype="float32", minimize_method='BFGS'):
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
    result = minimize(objective_ulp, initial_guess)

    return result.x.tolist()


def generate_function_from_poly(coeffs):
    return lambda x: np.polyval(coeffs, x)


def generate_approximations(fun, max_poly_rank, xrange, npoints, function_name="", dtype="float32"):

    xmin, xmax = xrange

    functions = {}

    # Numpy does not support bfloat16, so we use float32 instead as fallback
    if dtype == "bfloat16":
        dtype = "float32"

    # Test with exponential function
    for poly_rank in range(1, max_poly_rank + 1):
        # Generate Chebyshev polynomial coefficients for given rank
        cheby_coeffs = generate_poly_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype=dtype)
        cheby_fun = generate_function_from_poly(cheby_coeffs)
        functions[f'{function_name}-Chebyshev[{poly_rank}]'] = cheby_fun

        polynomial_coeffs = generate_polynomial_approx(fun, poly_rank, xmin, xmax, npoints, dtype=dtype, minimize_method='BFGS')
        poly_fun = generate_function_from_poly(polynomial_coeffs)
        functions[f'{function_name}-Minimize[{poly_rank}]'] = poly_fun

    return functions
