import torch
print("imported Torch")

n = 3
z = torch.normal(torch.zeros([3, 3]))
print(z)
z = torch.nn.functional.normalize(z, dim=-1)
print("done calculating")
print(z)