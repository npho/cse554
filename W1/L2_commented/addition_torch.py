import torch
from torch.profiler import profile, record_function, ProfilerActivity

def add_torch(a, b):
    return a + b

num = 100000000

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    
    a = torch.rand(num)
    b = torch.rand(num)

    a = a.to('cuda')
    b = b.to('cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(1000):
        result = add_torch(a, b)
    end.record()
    torch.cuda.synchronize()

    per_iteration_time = start.elapsed_time(end) / 1000
    per_iteration_time_second = per_iteration_time / 1000
    print(f"Time taken for 1 iterations: {per_iteration_time_second} seconds")
    print(f"Bandwidth: {3 * num * a.element_size() / per_iteration_time_second / 1e9} GB/s")

    result = result.to('cpu')

    print(result)

prof.export_chrome_trace("trace.json")