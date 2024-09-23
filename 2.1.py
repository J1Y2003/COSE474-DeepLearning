import torch

x = torch.arange(12, dtype=torch.float32)
X = x.reshape(-1,4)
#print(X)

print(torch.ones(2, 3, 4))