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
