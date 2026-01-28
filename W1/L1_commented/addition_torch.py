import torch

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

c = c.to('cpu')

print(c)
