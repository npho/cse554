import torch
from torch.profiler import profile, record_function, ProfilerActivity


def add_tensors(a, b):
    return a + b

num = 100000000
a = torch.rand(num, device='cpu')
b = torch.rand(num, device='cpu')

print(a)
print(b)

a = a.to('cuda')
b = b.to('cuda')

c = add_tensors(a, b)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    for _ in range(1000):
        c = add_tensors(a, b)
    
prof.export_chrome_trace("trace.json")

c = c.to('cpu')

print(c)

