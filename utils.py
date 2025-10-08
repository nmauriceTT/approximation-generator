import numpy as np
import torch



def reinterpret_bf16_to_u16(x):
    # Python does not use data types with excplicit data sizes. 
    # To manipulates fixed sizes, we leverage both torch and numpy
    # Each has its limitation for BF16 -> UINT16 reinterpretation
    # - torch has limited support for int16
    # - numpy has limited support for bfloat16
    # To bypass these limitations, we first cast data to bfloat16, and reinterpret to int16 using torch
    # Then we convert tensor to numpy, reinterpret its type to uint16 and get the value back to a python variable 
    torch_bf16 = torch.full([1], x, dtype=torch.bfloat16)
    torch_i16 = torch_bf16.view(torch.int16)
    np_u16 = torch_i16.numpy().view(np.uint16)

    return np_u16[0]

def reinterpret_bf16_to_i16(x):
    torch_bf16 = torch.full([1], x, dtype=torch.bfloat16)
    torch_i16 = torch_bf16.view(torch.int16)
    return torch_i16.item()

def reinterpret_u16_to_bf16(n):

    np_i16 = np.full([1], n, dtype=np.uint16).view(np.int16)
    torch_i16 = torch.from_numpy(np_i16)
    torch_bf16 = torch_i16.view(torch.bfloat16)

    return torch_bf16.item()

def reinterpret_i16_to_bf16(n):
    torch_i16 = torch.full([1], n, dtype=torch.int16)
    torch_bf16 = torch_i16.view(torch.bfloat16)
    return torch_bf16.item()


def forward_transformation(x):
    """
    Apply value-space transformation from bfloat16 to int16
    """
    value_i16 = reinterpret_bf16_to_i16(x)

    if value_i16 < 0:
        value_i16 ^= 0x7fff
        
    return value_i16

def backward_transformation(n):
    """
    Apply value-space transformation from int16 to bfloat16
    """
    if n < 0:
        n ^= 0x7fff

    return reinterpret_i16_to_bf16(n)




def gen_arange_nulp_bf16(start, end, ulp_step):
    """
    Generate a numpy array that contains `length` ordered bfloat16 numbers from `start`,
    with two consecutive numbers being separated by `ulp_step` ULPs.
    
    Args:
        length: Number of elements to generate
        start: Starting float value
        ulp_step: ULP (Units in the Last Place) step between consecutive numbers
        
    Returns:
        torch tensor of bfloat16 values
    """
    
    # Apply transformation (same as C++ code)
    # The Bfloat16 space is laid out as follows:
    # 0x0000        0x7f80        0x8000        0xff80        0xffff
    #  _________________________________________________________
    #  |  0 -> +inf  |   +inf-NaN  |  0 -> -inf  |   -inf-NaN  |
    #  _________________________________________________________
    #
    # We transform this 'value' space to the following:
    #
    # 0x0000        0x7f80        0x8000        0xff80        0xffff
    #  _________________________________________________________
    #  |  -inf-NaN  |   -inf -> 0  |  0 -> +inf  |   +inf-NaN  |
    #  _________________________________________________________
    #
    # This is achived by first 'permuting' first and second halves of the space
    # and then 'inverting' the new 'first half' so that values are in increasing order
    # 
    # Thank to this trick, the 'matching' floating-point number 
    # (after backward transformation) from 0x7f80 to 0xff80 will be in ascending order,
    # without ranges with non-finite values in-between

    # The forward step can be achieved as follows:
    # u = x ^ 0x8000 (swap halves of value space)
    # if u < 0x8000: # if x was negative
    #   u = u ^ 0x7fff # invert 'negative' value space to make it ascending
    #
    # The backward step can be achived with the following
    # if u <= 0x8000:
    #   u = u ^ 0x7fff
    # x = u ^ 0x8000

    if start > end:
        ulp_step = -abs(ulp_step)

    
    start_scalar = forward_transformation(start)
    end_scalar = forward_transformation(end)
    length = (end_scalar - start_scalar) // ulp_step + 1

    print(f"start scalar = {start_scalar} (bw = {backward_transformation(start_scalar)}). end scalar = {end_scalar} (bw = {backward_transformation(end_scalar)})")


    # Create [start, start + step, start + 2*step, ... , end] sequence
    torch_data = torch.arange(start_scalar, end_scalar, ulp_step, dtype=torch.int16)

    # Apply backward value-space transformation
    negative_mask = torch.less(torch_data, 0)
    torch_data[negative_mask] = torch_data[negative_mask] ^ int(0x7fff)

    torch_data_bf16 = torch_data.view(torch.bfloat16)

    return torch_data_bf16


def gen_linspace_nulp_bf16(start, end, num_elements, verbose=False):
    """
    Generate a torch array that contains bfloat16 numbers from `start` to `end`,
    with `num_elements` elements using ULP-based spacing.
    
    Args:
        start: Starting float value
        end: Ending float value
        num_elements: Number of elements to generate
        verbose: Whether to print debug information
        
    Returns:
        torch tensor of bfloat16 values from start to end (inclusive)
    """
    
    # Transform start and end to the transformed space
    start_scalar = forward_transformation(start)
    end_scalar = forward_transformation(end)
    
    # Calculate the ULP step needed to get approximately num_elements
    if num_elements <= 1:
        # For single element or invalid count, use a large ULP step
        ulp_step = 10000
        length = 1
    else:
        # Calculate ULP step to get the desired number of elements
        ulp_range = abs(end_scalar - start_scalar)
        ulp_step = max(1, ulp_range // (num_elements - 1))
        
        # Calculate actual length based on the ULP step
        if end_scalar >= start_scalar:
            length = (end_scalar - start_scalar) // ulp_step + 1
        else:
            # Handle case where end < start
            length = (start_scalar - end_scalar) // ulp_step + 1
            ulp_step = -ulp_step
    
    if verbose:
        print(f"start scalar = {start_scalar} (bw = {backward_transformation(start_scalar)}), end scalar = {end_scalar} (bw = {backward_transformation(end_scalar)})")
        print(f"requested num_elements = {num_elements}, calculated ulp_step = {ulp_step}, actual length = {length}")
    
    # Create sequence in transformed space
    if ulp_step > 0:
        torch_data = torch.arange(start_scalar, end_scalar, ulp_step, dtype=torch.int16)
    else:
        torch_data = torch.arange(start_scalar, end_scalar, ulp_step, dtype=torch.int16)
    
    # Apply backward value-space transformation
    negative_mask = torch.less(torch_data, 0)
    torch_data[negative_mask] = torch_data[negative_mask] ^ int(0x7fff)
    
    torch_data_bf16 = torch_data.view(torch.bfloat16)
    
    return torch_data_bf16
