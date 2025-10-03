import torch
import numpy as np



def ulp(x):
    x_next = torch.nextafter(x, torch.tensor([float('inf')], dtype=x.dtype))
    return x_next - x


def ulp_delta(calculated, golden):

    if isinstance(calculated, np.float64):
        calculated = torch.tensor(calculated, dtype=torch.float64)
    if isinstance(golden, np.float64):
        golden = torch.tensor(golden, dtype=torch.float64)

    if isinstance(calculated, np.ndarray):
        calculated = torch.tensor(calculated)
    if isinstance(golden, np.ndarray):
        golden = torch.tensor(golden)

    ulp_values = ulp(golden.to(calculated.dtype)).to(golden.dtype)
    abs_diff = torch.abs(calculated.to(golden.dtype) - golden).to(golden.dtype)
    return abs_diff / ulp_values


def worst_ulp_delta(calculated, golden):
    ulp_deltas = ulp_delta(calculated, golden)
    return torch.max(ulp_deltas)


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