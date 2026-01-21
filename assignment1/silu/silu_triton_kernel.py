# silu_triton_kernel.py
# CSE 554 Group 14
# Implementation of the SiLU activation function using Triton.

import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(x1_ptr, x2_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for SiLU activation function.

    Args:
        x1_ptr: Pointer to input vector.
        x2_ptr: Pointer to output vector.
        n_elements: Size of the vector.
        BLOCK_SIZE: Number of elements processed by each block.

    Returns:
        None
    """
    # All indices and offset calculations
    pid = tl.program_id(axis=0) # block index equivalent
    block_start = pid * BLOCK_SIZE # block starting offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # elements in block

    # Load input values from DRAM, mask out of-bounds accesses
    mask = offsets < n_elements # req'd if input isn't multiple of BLOCK_SIZE
    x1 = tl.load(x1_ptr + offsets, mask=mask)

    # Calculations on GPU
    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    x2 = x1 * tl.sigmoid(x1)

    # Write results back to DRAM
    tl.store(x2_ptr + offsets, x2, mask=mask)

def silu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the SiLU Triton kernel.

    Args:
        x: Input tensor.

    Returns:
        Output tensor with SiLU activation applied.
    """

    # Ensure input tensor is on GPU and in contiguous memory
    if not x.is_cuda:
        x = x.to("cuda")

    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x) # Allocate result tensor
    n_elements = x.numel()

    # Calculate grid size (number of blocks needed)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Launch the kernel and return result
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
