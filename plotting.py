import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility import ulp_delta


def plot_approximation_ulp_error(approximation_funcs, xrange, golden_function, dtype=torch.float32, filename="ulp_error_plot.pdf", npoints=1000, plot_params={}):
    """
    Plot ULP error scatter plot for approximations against their golden function.

    Args:
        approximation_funcs: Dictionary of {name: function} where function is a callable that computes the approximation
        xrange: Tuple (xmin, xmax) defining the x-axis range
        golden_function: The golden function to compare against (required for ULP calculation)
        dtype: Data type for approximation calculations (default: torch.float32)
        filename: Output PDF filename (default: "ulp_error_plot.pdf")
        npoints: Number of points to plot (default: 1000)
        plot_params: Dictionary of plot parameters (xlim, ylim, xscale, yscale)
    """
    if golden_function is None:
        raise ValueError("Golden function is required for ULP error calculation")

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    torch_dtype = getattr(torch, dtype)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate x values
    xmin, xmax = xrange
    x_values = np.linspace(xmin, xmax, npoints)

    # Compute golden values once
    golden_values = torch.tensor([golden_function(x) for x in x_values], dtype=torch.float64) # Keep as float64 for fractional ULP error calculation

    # Prepare data for seaborn plotting
    colors = sns.color_palette("husl", len(approximation_funcs))

    all_approx_data = []
    for i, (name, approx_func) in enumerate(approximation_funcs.items()):
        # Compute approximation values
        approx_values = torch.tensor([approx_func(x) for x in x_values], dtype=torch_dtype)

        # Compute ULP errors for each point
        ulp_errors = ulp_delta(approx_values, golden_values).numpy()
        # print(f"ulp_errors: {ulp_errors}")

        max_ulp_error = np.max(ulp_errors)

        series_name = [f'{name} (max: {max_ulp_error:.2f} ULP)'] * len(x_values)
        series_color = [colors[i]] * len(x_values)

        all_approx_data += [pd.DataFrame(
            {
                'x': x_values,
                'ulp_error': ulp_errors,
                'approximation': series_name,
                'color': series_color
            }
        )]

    # Convert to DataFrame for seaborn
    df = pd.concat(all_approx_data, axis=0)
    
    # Create scatter plot using seaborn
    ax = sns.scatterplot(data=df, x='x', y='ulp_error', hue='approximation',
                        alpha=1, s=10, ax=ax)

    # Customize plot using ax methods
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('ULP Error')
    ax.set_title('ULP Error Scatter Plot for Approximations')

    ax.set_xlim(plot_params.get('xlim', None))
    ax.set_ylim(plot_params.get('ylim', None))
    ax.set_xscale(plot_params.get('xscale', 'linear'))
    ax.set_yscale(plot_params.get('yscale', 'linear'))

    # Use seaborn to improve the overall appearance
    sns.despine(ax=ax)

    # Save as PDF
    fig.tight_layout()
    fig.savefig(f"{filename}.png", format='png', bbox_inches='tight', dpi=300)
    # fig.savefig(filename.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to prevent it from displaying


def plot_approximation(approximation_funcs, xrange, golden_function=None, filename="function-plot", npoints=1000, plot_params={}):
    """
    Plot approximations against their golden function.

    Args:
        approximation_funcs: Dictionary of {name: function} where function is a callable that computes the approximation
        xrange: Tuple (xmin, xmax) defining the x-axis range
        golden_function: The golden function to compare against (optional)
        filename: Output PDF filename (default: "approximation_plot.pdf")
        npoints: Number of points to plot (default: 1000)
        plot_params: Dictionary of plot parameters (xlim, ylim, xscale, yscale)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate x values
    xmin, xmax = xrange
    x_values = np.linspace(xmin, xmax, npoints)

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
    for name, approx_func in approximation_funcs.items():
        approx_values = [approx_func(x) for x in x_values]

        for x, y in zip(x_values, approx_values):
            plot_data.append({
                'x': x,
                'y': y,
                'function': name
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
    ax.set_title('Approximations vs Golden Function')

    ax.set_xlim(plot_params.get('xlim', None))
    ax.set_ylim(plot_params.get('ylim', None))
    ax.set_xscale(plot_params.get('xscale', 'linear'))
    ax.set_yscale(plot_params.get('yscale', 'linear'))

    # Use seaborn to improve the overall appearance
    sns.despine(ax=ax)

    # Save as PDF
    fig.tight_layout()
    fig.savefig(f"{filename}.png", format='png', bbox_inches='tight', dpi=300)
    # fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to prevent it from displaying
