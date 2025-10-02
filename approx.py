import math
import os

from approximation import generate_approximations
from plotting import plot_approximation, plot_approximation_ulp_error





def generate_and_plots(fun, npoints, xrange, dtype, fun_name, approx_plot_params={}, ulp_error_plot_params={}, extra_approximations={}):

    output_dir = "plots/"    
    os.makedirs(output_dir, exist_ok=True)

    approx_functions = generate_approximations(fun, max_poly_rank=8, xrange=xrange, npoints=npoints, function_name=fun_name, dtype=dtype)
    approx_functions.update(extra_approximations)

    plot_approximation_ulp_error(approx_functions, xrange, fun, dtype, filename=f"{output_dir}/{fun_name}[{dtype}]_ulp_error.pdf", plot_params=ulp_error_plot_params)




# Example usage and testing
if __name__ == "__main__":
    # Test with exponential function
    print("Testing polynomial approximation of exp(x) on [0, 1]")
    
    npoints = 10

    tanh_approximations = {
        "tanh-exp": lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)),
        "tanh-exp2": lambda x: (2**(x / math.log(2)) - 2**(-x / math.log(2))) / (2**(x / math.log(2)) + 2**(-x / math.log(2))),
    }



    generate_and_plots(math.exp, npoints, (0, 1), "bfloat16", "exp", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 20)})
    generate_and_plots(math.atan, npoints, (-10, 10), "bfloat16", "atan", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_and_plots(math.asin, npoints, (0, 1), "bfloat16", "asin", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 40)})
    generate_and_plots(lambda x: math.pow(2, x), npoints, (0, 1), "bfloat16", "exp2", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_and_plots(lambda x: math.log(x + 1), npoints, (0, 10), "bfloat16", "log1p", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 10)})
    generate_and_plots(lambda x: math.log(x + 1), npoints, (0, 10), "bfloat16", "log1p", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 100)})
    generate_and_plots(math.tanh, npoints, (0, 10), "bfloat16", "tanh", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 100)}, extra_approximations=tanh_approximations)

    generate_and_plots(math.exp, npoints, (0, 1), "float32", "exp", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots(math.atan, npoints, (-10, 10), "float32", "atan", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots(math.asin, npoints, (0, 1), "float32", "asin", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots(lambda x: math.pow(2, x), npoints, (0, 1), "float32", "exp2", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots(lambda x: math.log(x + 1), npoints, (0, 10), "float32", "log1p", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)})
    generate_and_plots(math.tanh, npoints, (0, 10), "float32", "tanh", approx_plot_params={}, ulp_error_plot_params={'ylim': (0, 1000)}, extra_approximations=tanh_approximations)
