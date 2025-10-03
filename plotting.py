import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility import ulp_delta


def plot_approximation_ulp_error(detailed_df, xrange=None, filename="ulp_error_plot.pdf", plot_params={}):
    """
    Plot ULP error scatter plot for approximations against their golden function.

    Args:
        detailed_df: DataFrame with columns: function_name, approx_name, datatype, input_value, output_value, ulp_error (preferred)
        approximation_funcs: Dictionary of {name: function} (legacy mode, used if detailed_df is None)
        xrange: Tuple (xmin, xmax) defining the x-axis range (legacy mode)
        golden_function: The golden function to compare against (legacy mode)
        dtype: Data type for approximation calculations (legacy mode)
        filename: Output PDF filename (default: "ulp_error_plot.pdf")
        npoints: Number of points to plot (default: 1000, used only in legacy mode)
        plot_params: Dictionary of plot parameters (xlim, ylim, xscale, yscale)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # New mode: use detailed DataFrame
    # Filter out golden function from detailed data
    df_approx = detailed_df[detailed_df['function_name'] != 'golden'].copy()

    # Get unique approximation names
    approx_names = df_approx['approx_name'].unique()
    colors = sns.color_palette("husl", len(approx_names))

    # Calculate max ULP error for each approximation and add to labels
    all_approx_data = []
    for i, name in enumerate(approx_names):
        df_func = df_approx[df_approx['approx_name'] == name]
        max_ulp_error = df_func['ulp_error'].max()

        df_func_plot = df_func.copy()
        df_func_plot['approximation'] = f'{name} (max: {max_ulp_error:.2f} ULP)'
        df_func_plot['x'] = df_func_plot['input_value']

        all_approx_data.append(df_func_plot)

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


def plot_approximation(detailed_df, xrange=None, filename="plot.pdf", plot_params={}):
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

    # Prepare data for seaborn plotting
    plot_data = []

    # Create line plot using seaborn
    ax = sns.lineplot(data=detailed_df, x='input_value', y='output_value', hue='approx_name', ax=ax)

    # Customize the golden function line to be more prominent
    if 'golden' in detailed_df['approx_name'].unique():
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
