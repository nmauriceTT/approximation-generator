import torch
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utility import ulp_delta, worst_ulp_delta, compare_ulp_error, gen_arange_nulp_bf16
import math

import mpmath as mp



def generate_polynomial_chebyshev(fun, degrees: int, xmin, xmax, npoints, dtype="float32"):
    np_dtype = getattr(np, dtype)

    npoints = 1000
    np_inputs = np.linspace(xmin, xmax, npoints)
    np_outputs = np.vectorize(fun)(np_inputs)

    # Determine data dtype resolution
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

    print(f"degrees: {degrees}")
    chebyshev_coeffs = np.polynomial.chebyshev.chebfit(np_inputs, np_outputs, deg=degrees, rcond=rcond)

    
    # Revert coefficients for consistentcy with np.polyval
    coeffs = np.flip(np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs))

    return coeffs.astype(np_dtype).tolist()

def generate_polynomial_chebyshev_bis(fun, deg: int, xmin, xmax, npoints, dtype="float32"):
    np_dtype = getattr(np, dtype)


    print(f"==== degrees: {deg}")
    chebyshev_coeffs = np.polynomial.chebyshev.Chebyshev.interpolate(lambda x: np.vectorize(fun)(x), deg=deg, domain=(xmin, xmax))
    print(f"chebyshev_coeffs: {chebyshev_coeffs.coef}")
    # chebyshev_coeffs.coef[0] = 1
    
    poly_coeffs = chebyshev_coeffs.convert(kind=np.polynomial.Polynomial)

    np_inputs = np.array([0.0, 0.5, 1.0])
    np_outputs = chebyshev_coeffs(np_inputs)
    np_poly_output = poly_coeffs(np_inputs)
    print(f"np_outputs: {np_outputs}")
    print(f"np_poly_output: {np_poly_output}")

    coeffs = np.flip(poly_coeffs.coef)
    # Revert coefficients for consistentcy with np.polyval
    # coeffs = np.flip(np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs))
    # coeffs = np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs.coef)
    # coeffs = np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs.coef)
    

    print(f"coeffs: {coeffs}")

    np_poly_output = np.polyval(coeffs, np_inputs)
    print(f"np_poly_output: {np_poly_output}")
    print()

    return coeffs.astype(np_dtype).tolist()

def generate_polynomial_chebyshev_shifted(fun, deg: int, xmin, xmax, npoints, dtype="float32"):

    shifted_fun = lambda x: (fun(x) - 1)/x

    if xmin == 0:
        xmin = max(xmin, 1e-9) # Shift xmin improve numeric stability

    polynomial_coeffs = generate_polynomial_chebyshev_bis(shifted_fun, max(0, deg - 1), xmin, xmax, npoints, dtype)
    
    polynomial_coeffs = np.concatenate([polynomial_coeffs, [fun(0)]])
    return polynomial_coeffs


def generate_polynomial_chebyshev_mp(fun, deg: int, xmin, xmax, dtype="float32"):
    np_dtype = getattr(np, dtype)

    
    chebyshev_coeffs = mp.chebyfit(fun, interval=(xmin, xmax), N=deg)
    # Revert coefficients for consistentcy with np.polyval
    # coeffs = np.flip(np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs))
    
    return np.array(chebyshev_coeffs).astype(np_dtype).tolist()



def minimize_objective():
    pass


# Polynomial approximation using scipy's minimize function
def generate_polynomial_approx(fun, deg: int, xmin, xmax, npoints, dtype="float32", minimize_method='BFGS', function_derivative=None):
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


    # np_x_points = np.linspace(xmin, xmax, npoints)

    torch_x_points = gen_arange_nulp_bf16(xmin, xmax, ulp_step=2).to(torch_dtype)
    np_x_points = torch_x_points.to(torch.float64).numpy()


    np_y_golden_points = np.vectorize(fun)(np_x_points)

    def compute_gradient(x):
        if function_derivative is None:
            return None
        return np.vectorize(function_derivative)(x)

    def objective(coeffs):
        """Objective function: maximum relative error"""

        np_polyval = np.polyval(coeffs, np_x_points)
        torch_calculated = torch.tensor(np_polyval, dtype=torch_dtype)

        abs_error = abs(torch_calculated - torch.tensor(np_y_golden_points, dtype=torch_dtype))
        relative_errors = abs_error / abs(torch.tensor(np_y_golden_points, dtype=torch_dtype))

        return torch.max(abs_error ** 2)

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
    initial_guess = np.zeros(deg)

    # Use least squares as initial approximation for better starting point
    # Build Vandermonde matrix for least squares
    A = np.vander(np_x_points, deg, increasing=True)
    initial_guess = np.linalg.lstsq(A, np_y_golden_points, rcond=None)[0]

    # Minimize the maximum relative error
    result = minimize(objective_ulp, initial_guess, method=minimize_method, jac=compute_gradient)

    result_coeffs = np.flip(result.x)
    return result_coeffs.tolist()


def generate_function_from_poly(coeffs):
    return lambda x: np.polyval(coeffs, x)


class GenericApproximation:

    def __init__(self, name, lambda_func, dtype):
        self.name = name
        self.type = "lambda"
        self.lambda_func = lambda_func
        self.dtype = dtype
    
    def __call__(self, x):
        return self.lambda_func(x)

    def __str__(self):
        return f"{self.name}[{self.dtype}]"

    def serialize(self):
        return ""

class Approximation:

    def __init__(self, name, type, poly_coeffs, dtype):
        self.coeff = poly_coeffs
        self.name = name
        self.type = type
        self.rank = len(poly_coeffs)
        self.dtype = dtype

    def __call__(self, x):
        return np.polyval(self.coeff, x)

    def __str__(self):
        return f"{self.name}-{self.rank}[{self.dtype}]"

    def serialize(self):
        coeff_str = ",".join([str(coeff) for coeff in self.coeff])
        coeff_str = f"{self.name}(x) = {coeff_str}"
        return coeff_str


def generate_approximations(fun, max_poly_rank, xrange, npoints, function_name="", dtype="float32", function_derivative=None) -> dict[str, Approximation]:

    xmin, xmax = xrange

    functions: dict[str, Approximation] = {}

    # Numpy does not support bfloat16, so we use float32 instead as fallback
    if dtype == "bfloat16":
        dtype = "float32"

    # Test with exponential function
    for poly_rank in range(1, max_poly_rank + 1):
        # Generate Chebyshev polynomial coefficients for given rank
        # cheby_coeffs = generate_polynomial_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype=dtype)

        cheby_coeffs = generate_polynomial_chebyshev(fun, poly_rank, xmin, xmax, npoints, dtype=dtype)
        cheby_fun = Approximation(f'{function_name}-Chebyshev-v0[{poly_rank}]', 'chebyshev', cheby_coeffs, dtype)
        functions[cheby_fun.name] = cheby_fun

        cheby_coeffs = generate_polynomial_chebyshev_bis(fun, poly_rank, xmin, xmax, npoints, dtype=dtype)
        cheby_fun = Approximation(f'{function_name}-Chebyshev-v1[{poly_rank}]', 'chebyshev', cheby_coeffs, dtype)
        functions[cheby_fun.name] = cheby_fun

        cheby_coeffs = generate_polynomial_chebyshev_mp(fun, poly_rank, xmin, xmax, dtype=dtype)
        cheby_fun = Approximation(f'{function_name}-Chebyshev-v2[{poly_rank}]', 'chebyshev', cheby_coeffs, dtype)
        functions[cheby_fun.name] = cheby_fun

        cheby_coeffs = generate_polynomial_chebyshev_shifted(fun, poly_rank, xmin, xmax, npoints, dtype=dtype)
        cheby_fun = Approximation(f'{function_name}-Chebyshev-v1-c0ef0[{poly_rank}]', 'chebyshev', cheby_coeffs, dtype)
        functions[cheby_fun.name] = cheby_fun

        polynomial_coeffs = generate_polynomial_approx(fun, poly_rank+1, xmin, xmax, npoints, dtype=dtype, minimize_method='L-BFGS-B', function_derivative=function_derivative)
        poly_fun = Approximation(f'{function_name}-Minimize[{poly_rank}]', 'minimize', polynomial_coeffs, dtype)
        functions[poly_fun.name] = poly_fun

    return functions
