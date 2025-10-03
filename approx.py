import math
import os
import pandas as pd

from approximation import generate_approximations, GenericApproximation
from plotting import plot_approximation, plot_approximation_ulp_error
from measurement import measure_approximation_error


FUNCTIONS = {
    "exp": {
        "fun":math.exp,
        "derivative": math.exp,
    },
    "atan": {
        "fun": math.atan,
        "derivative": lambda x:1 / (1 + x**2),
    },
    "asin": {
        "fun": math.asin,
        "derivative": lambda x: 1 / math.sqrt(1 - x**2),
    },
    "exp2": {
        "fun": lambda x: math.pow(2, x),
        "derivative": lambda x: math.pow(2, x) * math.log(2),
    },
    "log1p": {
        "fun": lambda x: math.log(x + 1),
        "derivative": lambda x: 1 / (x + 1),
    },
    "tanh": {
        "fun": math.tanh,
        "derivative": lambda x: 1 - math.tanh(x) ** 2,
    },
}


def generate_and_plots(fun_name, npoints, xrange, dtype, approx_plot_params={}, ulp_error_plot_params={}, extra_approximations={}):

    output_dir = "plots/"    
    os.makedirs(output_dir, exist_ok=True)

    fun = FUNCTIONS[fun_name]["fun"]
    derivative = FUNCTIONS[fun_name]["derivative"] if "derivative" in FUNCTIONS[fun_name] else None


    approx_functions = generate_approximations(fun, max_poly_rank=8, xrange=xrange, npoints=npoints, function_name=fun_name, dtype=dtype, function_derivative=derivative)
    approx_functions.update(extra_approximations)


    output_dir = f"{output_dir}/{fun_name}/"
    os.makedirs(output_dir, exist_ok=True)

    all_approx_types = set([approx.type for approx in approx_functions.values()])

    filename = f"{fun_name}[{dtype}]"


    (summary_df, detailed_df) = measure_approximation_error(fun_name, approx_functions, fun, xrange, dtype, npoints)

    summary_df.to_csv(f"{output_dir}/{filename}.csv", index=False)

    plot_approximation_ulp_error(detailed_df, xrange, filename=f"{output_dir}/{filename}-ulp", plot_params=ulp_error_plot_params)
    plot_approximation(detailed_df, xrange, filename=f"{output_dir}/{filename}", plot_params=approx_plot_params)

    for approx_type in all_approx_types:

        subdir = f"{output_dir}/{approx_type}/"
        os.makedirs(subdir, exist_ok=True)

        detailed_df_type = detailed_df[detailed_df['approx_type'] == approx_type]

        plot_approximation_ulp_error(detailed_df_type, xrange, filename=f"{subdir}/{filename}-ulp", plot_params=ulp_error_plot_params)
        plot_approximation(detailed_df_type, xrange, filename=f"{subdir}/{filename}", plot_params=approx_plot_params)


# Example usage and testing
if __name__ == "__main__":
    # Test with exponential function
    print("Testing polynomial approximation of exp(x) on [0, 1]")
    
    npoints = 1000

    tanh_approximations = {
        "tanh-exp": GenericApproximation("tanh-exp", lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), "bfloat16"),
        "tanh-exp2": GenericApproximation("tanh-exp2", lambda x: (2**(x / math.log(2)) - 2**(-x / math.log(2))) / (2**(x / math.log(2)) + 2**(-x / math.log(2))), "bfloat16"),
    }

    generate_and_plots("exp", npoints, (0, 1), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 20)})
    generate_and_plots("atan", npoints, (-10, 10), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    # generate_and_plots("asin", npoints, (0, math.nextafter(1, -math.inf)), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_and_plots("exp2", npoints, (0, 1), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_and_plots("log1p", npoints, (math.nextafter(0, math.inf), 10), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_and_plots("tanh", npoints, (0, 4), "bfloat16", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 100)}, extra_approximations=tanh_approximations)

    generate_and_plots("exp", npoints, (0, 1), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots("atan", npoints, (-10, 10), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    # generate_and_plots("asin", npoints, (0, math.nextafter(1, -math.inf)), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots("exp2", npoints, (0, 1), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots("log1p", npoints, (math.nextafter(0, math.inf), 10), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots("tanh", npoints, (0, 4), "float32", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)}, extra_approximations=tanh_approximations)
