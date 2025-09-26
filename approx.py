import torch
import math
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


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
    result = minimize(objective_ulp, initial_guess, method=minimize_method)
    
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
    def polynomial_eval(x):
        return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
    
    approx_values = torch.tensor([polynomial_eval(x) for x in x_points], dtype=dtype)
    
    # Compute worst ULP delta
    return worst_ulp_delta(approx_values, golden_values)



def generate_polynomial_approx(fun, poly_rank, xrange, npoints, dtype="float32", minimize_method='BFGS'):

    xmin, xmax = xrange

    # Test with exponential function
    coeffs = build_polynomial_approx(fun, poly_rank, xmin, xmax, npoints, dtype="float32", minimize_method=minimize_method)
    print(f"rank {poly_rank} - Polynomial coefficients: {coeffs}")

    # Test ULP error
    ulp_error = compare_ulp_error(fun, coeffs, xmin, xmax, npoints)
    print(f"    Maximum ULP error: {ulp_error}")
    
    return coeffs


def plot_approximation_ulp_error(approximations, xrange, polynomial_ranks, golden_function=None, dtype=torch.float32, filename="ulp_error_plot.pdf", npoints=1000, plot_params={}):
    """
    Plot ULP error scatter plot for polynomial approximations against their golden function.
    
    Args:
        approximations: List of polynomial coefficient lists [approx0, approx1, ...]
        xrange: Tuple (xmin, xmax) defining the x-axis range
        polynomial_ranks: List of polynomial ranks corresponding to each approximation
        golden_function: The golden function to compare against (required for ULP calculation)
        dtype: Data type for approximation calculations (default: torch.float32)
        filename: Output PDF filename (default: "ulp_error_plot.pdf")
        npoints: Number of points to plot (default: 1000)
    """
    if golden_function is None:
        raise ValueError("Golden function is required for ULP error calculation")
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate x values
    xmin, xmax = xrange
    x_values = np.linspace(xmin, xmax, npoints)
    
    # Define polynomial evaluation function
    def polynomial_eval(coeffs, x):
        """Evaluate polynomial with given coefficients at x"""
        return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
    
    # Compute golden values once
    golden_values = torch.tensor([golden_function(x) for x in x_values], dtype=torch.float64)
    
    # Prepare data for seaborn plotting
    plot_data = []
    colors = sns.color_palette("husl", len(approximations))
    
    for i, (approx_coeffs, rank) in enumerate(zip(approximations, polynomial_ranks)):
        # Compute approximation values
        approx_values = torch.tensor([polynomial_eval(approx_coeffs, x) for x in x_values], dtype=dtype)
        
        # Compute ULP errors for each point
        ulp_errors = ulp_delta(approx_values, golden_values).numpy()
        
        # Add data points for this approximation
        for x, ulp_err in zip(x_values, ulp_errors):
            plot_data.append({
                'x': x,
                'ulp_error': ulp_err,
                'rank': f'Rank {rank} (max: {np.max(ulp_errors):.2f} ULP)',
                'color': colors[i]
            })
    
    # Convert to DataFrame for seaborn
    df = pd.DataFrame(plot_data)
    
    # Create scatter plot using seaborn
    ax = sns.scatterplot(data=df, x='x', y='ulp_error', hue='rank', 
                        alpha=0.6, s=8, ax=ax)
    
    # Customize plot using ax methods
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('ULP Error')
    ax.set_title('ULP Error Scatter Plot for Polynomial Approximations')
    
    ax.set_xlim(plot_params.get('xlim', None))
    ax.set_ylim(plot_params.get('ylim', None))
    ax.set_xscale(plot_params.get('xscale', 'linear'))
    ax.set_yscale(plot_params.get('yscale', 'linear'))

    # Use seaborn to improve the overall appearance
    sns.despine(ax=ax)
    
    # Save as PDF
    fig.tight_layout()
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to prevent it from displaying
    
    print(f"ULP error plot saved as {filename}")

def plot_approximation(approximations, xrange, polynomial_ranks, golden_function=None, dtype=torch.float32, filename="approximation_plot.pdf", npoints=1000, plot_params={}):
    """
    Plot polynomial approximations against their golden function.
    
    Args:
        approximations: List of polynomial coefficient lists [approx0, approx1, ...]
        xrange: Tuple (xmin, xmax) defining the x-axis range
        polynomial_ranks: List of polynomial ranks corresponding to each approximation
        golden_function: The golden function to compare against (optional)
        dtype: Data type for approximation calculations (default: torch.float32)
        filename: Output PDF filename (default: "approximation_plot.pdf")
        npoints: Number of points to plot (default: 1000)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values
    xmin, xmax = xrange
    x_values = np.linspace(xmin, xmax, npoints)
    
    # Define polynomial evaluation function
    def polynomial_eval(coeffs, x):
        """Evaluate polynomial with given coefficients at x"""
        return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
    
    # Prepare data for seaborn plotting
    plot_data = []
    
    # Add golden function if provided
    if golden_function is not None:
        golden_values = [golden_function(x) for x in x_values]
        for x, y in zip(x_values, golden_values):
            plot_data.append({
                'x': x,
                'y': y,
                'function': 'Golden Function'
            })
    
    # Add each approximation
    for i, (approx_coeffs, rank) in enumerate(zip(approximations, polynomial_ranks)):
        approx_values = [polynomial_eval(approx_coeffs, x) for x in x_values]
        for x, y in zip(x_values, approx_values):
            plot_data.append({
                'x': x,
                'y': y,
                'function': f'Polynomial Rank {rank}'
            })
    
    # Convert to DataFrame for seaborn
    df = pd.DataFrame(plot_data)
    
    # Create line plot using seaborn
    ax = sns.lineplot(data=df, x='x', y='y', hue='function', ax=ax)
    
    # Customize the golden function line to be more prominent
    if golden_function is not None:
        # Find the golden function line and make it thicker and black
        for line in ax.lines:
            if line.get_label() == 'Golden Function':
                line.set_color('black')
                line.set_linewidth(2)
                line.set_alpha(0.8)
                break
    
    # Customize plot using ax methods
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Output (y)')
    ax.set_title('Polynomial Approximations vs Golden Function')
    
    ax.set_xlim(plot_params.get('xlim', None))
    ax.set_ylim(plot_params.get('ylim', None))
    ax.set_xscale(plot_params.get('xscale', 'linear'))
    ax.set_yscale(plot_params.get('yscale', 'linear'))


    # Use seaborn to improve the overall appearance
    sns.despine(ax=ax)
    
    # Save as PDF
    fig.tight_layout()
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to prevent it from displaying
    
    print(f"Plot saved as {filename}")

def generate_approximations(fun, fun_name, npoints, xrange, dtype, approx_plot_params={}, ulp_error_plot_params={}):

    print(f"Generating approximations for {fun_name} on {xrange} with {npoints} points and {dtype} dtype")

    # Build approximations with different ranks
    ranks = [2, 4, 6, 8]
    
    coeffs = [generate_polynomial_approx(fun, poly_rank=rank, xrange=xrange, npoints=npoints, dtype=dtype, minimize_method='BFGS') for rank in ranks]
    
    # Plot the approximations
    plot_approximation(
        approximations=coeffs,
        xrange=xrange,
        polynomial_ranks=ranks,
        golden_function=fun,
        dtype=dtype,
        filename=f"{fun_name}_approximation.pdf",
        plot_params=approx_plot_params
    )
    
    # Plot ULP error scatter plot
    plot_approximation_ulp_error(
        approximations=coeffs,
        xrange=xrange,
        polynomial_ranks=ranks,
        golden_function=fun,
        dtype=dtype,
        filename=f"{fun_name}_ulp_error.pdf",
        plot_params=ulp_error_plot_params
    )

    print()

# Example usage and testing
if __name__ == "__main__":
    # Test with exponential function
    print("Testing polynomial approximation of exp(x) on [0, 1]")
    
    npoints = 500

    generate_approximations(math.exp, "exp[bfloat16]", npoints, (0, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 20)})
    generate_approximations(math.atan, "atan[bfloat16]", npoints, (-10, 10), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_approximations(math.asin, "asin[bfloat16]", npoints, (-1, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_approximations(lambda x: math.pow(2, x), "exp2[bfloat16]", npoints, (0, 1), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_approximations(lambda x: math.log(x + 1), "log1p[bfloat16]", npoints, (0, 10), torch.bfloat16, approx_plot_params={}, ulp_error_plot_params={'ylim': (-1, 10)})
    

    generate_approximations(math.exp, "exp", npoints, (0, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(math.atan, "atan", npoints, (-10, 10), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(math.asin, "asin", npoints, (-1, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(lambda x: math.pow(2, x), "exp2", npoints, (0, 1), torch.float32, approx_plot_params={}, ulp_error_plot_params={})
    generate_approximations(lambda x: math.log(x + 1), "log1p", npoints, (0, 10), torch.float32, approx_plot_params={}, ulp_error_plot_params={})

