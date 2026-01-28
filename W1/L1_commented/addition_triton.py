import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x,
    y,
    output,
    num,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    
    load_range = start + tl.arange(0, BLOCK_SIZE)
    mask = load_range < num
    
    x_local = tl.load(x + load_range, mask=mask)
    y_local = tl.load(y + load_range, mask=mask)
    output_local = x_local + y_local + 2 + x_local * y_local * y_local
    
    tl.store(output + load_range, output_local, mask=mask)
    
    
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    numel = x.numel()
    BLOCK_SIZE = 1024
    
    output = torch.empty_like(x)
    
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    add_kernel[grid](x, y, output, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


num = 100000000

a = torch.rand(num, device='cuda')
b = torch.rand(num, device='cuda')
print(a)
print(b)

c = add(a, b)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for i in range(100):
    c = add(a, b)
end.record()
torch.cuda.synchronize()

iteration_time = start.elapsed_time(end) / 100 / 1000
print(f"Average time per iteration: {iteration_time:.6f} seconds")
print("Bandwidth: {:.2f} GB/s".format(3*num * a.element_size() * 1e-9 / iteration_time))

print(c)
