import torch
import numpy as np
import pandas as pd
import mpmath
from utility import ulp_delta, gen_arange_nulp_bf16


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
    input_tensor = gen_arange_nulp_bf16(0, 1, ulp_step=1).to(torch_dtype)

    np_input = input_tensor.to(torch.float64).numpy()
    npoints = np_input.size

    # Compute golden values once
    np_golden_tensor = np.vectorize(golden_function)(input_tensor.to(torch.float64).numpy())
    golden_tensor = torch.tensor(np_golden_tensor)

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
        'input_value': np_input,
        'output_value': np_golden_tensor,
        'ulp_error': 0.0,
        'formula': 'golden'
    })
    all_results.append(golden_results_df)

    for name, approx_func in approximation_funcs.items():
        # Compute approximation values

        tensor = input_tensor.clone()
        approx_tensor = tensor.apply_(approx_func)

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
            "input_value": np_input,
            "output_value": approx_tensor.to(torch.float64).numpy(),
            "ulp_error": ulp_errors
        })
        all_results.append(results_df)

    # detailed_df = pd.concat(all_results, ignore_index=True) if len(all_results) > 0 else pd.DataFrame(columns=all_results[0].columns)
    detailed_df = pd.concat(all_results, axis=0, ignore_index=True)

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_data)

    return summary_df, detailed_df
