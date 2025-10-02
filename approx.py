import torch
import math
import numpy as np
import os

from approximation import generate_polynomial_approx
from plotting import plot_approximation, plot_approximation_ulp_error

def generate_approximations(fun, fun_name, npoints, xrange, dtype, approx_plot_params={}, ulp_error_plot_params={}):

    print(f"Generating approximations for {fun_name} on {xrange} with {npoints} points and {dtype} dtype")

    # Build approximations with different ranks
    ranks = [2, 4, 6, 8]

    coeffs = [generate_polynomial_approx(fun, poly_rank=rank, xrange=xrange, npoints=npoints, dtype=dtype, minimize_method='BFGS') for rank in ranks]

    # Build dictionary of approximation functions for plotting
    approximation_funcs = {}
    for rank, coeff in zip(ranks, coeffs):
        approximation_funcs[f'Rank {rank}'] = lambda x, c=coeff: np.polyval(c, x)

    # Plot the approximations
    # plot_approximation(
    #     approximation_funcs=approximation_funcs,
    #     xrange=xrange,
    #     golden_function=fun,
    #     filename=f"{fun_name}_approximation.pdf",
    #     plot_params=approx_plot_params
    # )

    output_dir = "plots/"    
    os.makedirs(output_dir, exist_ok=True)

    # Plot ULP error scatter plot
    plot_approximation_ulp_error(
        approximation_funcs=approximation_funcs,
        xrange=xrange,
        golden_function=fun,
        dtype=dtype,
        filename=f"{output_dir}/{fun_name}_ulp_error.pdf",
        plot_params=ulp_error_plot_params
    )

    print()

# Example usage and testing
if __name__ == "__main__":
    # Test with exponential function
    print("Testing polynomial approximation of exp(x) on [0, 1]")
    
    npoints = 10

    generate_approximations(math.exp, "exp[bfloat16]", npoints, (0, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 20)})
    generate_approximations(math.atan, "atan[bfloat16]", npoints, (-10, 10), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_approximations(math.asin, "asin[bfloat16]", npoints, (0, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_approximations(lambda x: math.pow(2, x), "exp2[bfloat16]", npoints, (0, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_approximations(lambda x: math.log(x + 1), "log1p[bfloat16]", npoints, (0, 10), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    

    generate_approximations(math.exp, "exp", npoints, (0, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(math.atan, "atan", npoints, (-10, 10), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(math.asin, "asin", npoints, (-1, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(lambda x: math.pow(2, x), "exp2", npoints, (0, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(lambda x: math.log(x + 1), "log1p", npoints, (0, 10), torch.float32, approx_plot_params={}, ulp_error_plot_params={})

