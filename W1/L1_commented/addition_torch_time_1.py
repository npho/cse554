import torch
import time

def add_tensors(a, b):
    return a + b

num = 100000000
a = torch.rand(num, device='cpu')
b = torch.rand(num, device='cpu')

a = a.to('cuda')
b = b.to('cuda')

start_time = time.time()
for i in range(1000):
    result = add_tensors(a, b)
end_time = time.time()

per_iter_time = (end_time - start_time) / 1000
print("Time taken for 1000 iterations: ", end_time - start_time)
print("Time taken per 1 iteration: ", per_iter_time)

print("Bandwidth: ", (3* num * a.element_size()) / per_iter_time / 1e9, " GB/s")

result  = result.to('cpu')

print(result)
