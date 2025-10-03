import torch
import numpy as np
import pandas as pd
from utility import ulp_delta


def measure_approximation_error(function_name, approximation_funcs, golden_function, xrange, dtype, npoints=1000):
    """
    Measure ULP error for approximations against their golden function.

    Args:
        approximation_funcs: Dictionary of {name: function} where function is a callable that computes the approximation
        golden_function: The golden function to compare against (required for ULP calculation)
        xrange: Tuple (xmin, xmax) defining the input range
        dtype: Data type string for approximation calculations (e.g., "float32", "bfloat16")
        npoints: Number of points to measure (default: 1000)

    Returns:
        Tuple of two pandas DataFrames:
        - summary: N rows (one per function) with columns: function_name, approx_name, datatype, max_ulp_error, avg_ulp_error, median_ulp_error
        - detailed: (N × npoints) + golden rows with columns: function_name, approx_name, datatype, input_value, output_value, ulp_error
    """
    if golden_function is None:
        raise ValueError("Golden function is required for ULP error calculation")

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype)

    # Generate x values
    xmin, xmax = xrange
    x_values = np.linspace(xmin, xmax, npoints)

    # Compute golden values once
    golden_tensor = torch.tensor([golden_function(x) for x in x_values], dtype=torch.float64)

    # Prepare data structures
    summary_data = []

    # Process each approximation function
    all_results = []

    # Add golden function to detailed data
    golden_results_df  = pd.DataFrame.from_dict({
        'function_name': [function_name] * npoints,
        'approx_name': ['golden'] * npoints,
        'approx_type': ['golden'] * npoints,
        'datatype': [dtype] * npoints,
        'input_value': x_values,
        'output_value': golden_tensor.numpy(),
        'ulp_error': 0.0,
        'formula': 'golden'
    })
    all_results.append(golden_results_df)

    for name, approx_func in approximation_funcs.items():
        # Compute approximation values

        approx_values = np.vectorize(approx_func)(x_values)
        approx_tensor = torch.tensor(approx_values, dtype=torch_dtype)

        # Compute ULP errors for each point
        ulp_errors = ulp_delta(approx_tensor, golden_tensor).numpy()

        # Calculate summary statistics
        max_ulp_error = np.max(ulp_errors)
        avg_ulp_error = np.mean(ulp_errors)
        median_ulp_error = np.median(ulp_errors)

        # Add to summary data
        summary_data.append({
            'function_name': function_name,
            'approx_name': name,
            'approx_type': approx_func.type,
            'datatype': dtype,
            'max_ulp_error': max_ulp_error,
            'avg_ulp_error': avg_ulp_error,
            'median_ulp_error': median_ulp_error,
            'formula': approx_func.serialize()
        })

        results_df = pd.DataFrame.from_dict({
            "function_name": [function_name] * npoints,
            "approx_name": [name] * npoints,
            "approx_type": approx_func.type,
            "datatype": [dtype] * npoints,
            "input_value": x_values,
            "output_value": approx_values,
            "ulp_error": ulp_errors
        })
        all_results.append(results_df)

    # detailed_df = pd.concat(all_results, ignore_index=True) if len(all_results) > 0 else pd.DataFrame(columns=all_results[0].columns)
    detailed_df = pd.concat(all_results, axis=0, ignore_index=True)

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_data)

    return summary_df, detailed_df
