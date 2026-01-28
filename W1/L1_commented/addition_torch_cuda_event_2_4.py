import torch

def add_tensors(a, b):
    return a + b #+ 2 + a*b*b

num = 100000000
a = torch.rand(num, device='cpu')
b = torch.rand(num, device='cpu')

print(a)
print(b)

a = a.to('cuda')
b = b.to('cuda')

c = add_tensors(a, b)

# You need to remember to enable timing when creating the events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

# For torch and nsys we'll keep this at 1000 iterations, for ncu, we'll do 1 iteration
# since ncu has higher overhead due to being a more detailed profiler
for _ in range(1000):
    c = add_tensors(a, b)

end.record()
#end.synchronize()
torch.cuda.synchronize() # blocks the GPU until all queued GPU work is done 

per_iteration_time = start.elapsed_time(end) / 1000  / 1
print(f"Time per iteration: {per_iteration_time} ms")
print(f"Bandwidth: {3 * num * a.element_size() / per_iteration_time / 1e6} GB/s")

c = c.to('cpu')

print(c)
