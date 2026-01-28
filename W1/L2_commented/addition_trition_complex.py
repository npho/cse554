import torch
import triton
# This is a library of function that allow you to do things like data load/store and min/max
import triton.language as tl 

# This is the triton kernel
# This function will be called simultaneously by multiple blocks 
# tl.constexpr allows the triton compiler to bake BLOCK_SIZE into the kernel as a constant
# This allows for more optimizations (like loop unrolling)
@triton.jit
def add_kernel(x, y, output, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0) # this gives you the current block's id/index
    # Block boundaries:
    # block 0 -> 0 -> BLOCK_SIZE
    # block 1 -> BLOCK_SIZE -> 2 * BLOCK_SIZE

    block_start = block_idx * BLOCK_SIZE
    # Create an indexing vector and offset it by the block start
    offset_from_start = tl.arange(0, BLOCK_SIZE) + block_start
    # If the number of elements is not a multiple of BLOCK_SIZE,
    # We need to make sure we don't read/write out of bounds
    mask = offset_from_start < n_elements
    # Load will go to the registers unless it is large, in which case it will go to shared memory
    x_segment = tl.load(x + offset_from_start, mask=mask)
    y_segment = tl.load(y + offset_from_start, mask=mask)
    output_segment = x_segment + y_segment + x_segment * y_segment + x_segment * x_segment
    tl.store(output + offset_from_start, output_segment, mask=mask)

# Let's first write a wrapper CPU function that'll call the GPU function
# The wrapper will control grid/block size and launch the kernel, have error checking etc.
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Adds two tensors using Triton.
    """
    assert x.shape == y.shape

    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ) # wrapped in 1-tuple, since Triton expects a grid shape as a tuple
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

# We create the numbers on the GPU, 
x = torch.randn(100000000, device='cuda')
y = torch.randn(100000000, device='cuda')
output_triton = add(x, y)
output_torch = x + y + x*y + x*x

assert torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-4), "Triton addition does not match PyTorch addition!"

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    output_triton = add(x, y)
end.record()
torch.cuda.synchronize()

print(f"Triton addition time: {start.elapsed_time(end) / 100:.2f} ms")
print(f"Bandwidth: { 3 * x.numel() * x.element_size() / (start.elapsed_time(end) / 1000 / 100) / 1e9:.2f} GB/s")