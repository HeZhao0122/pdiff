import torch


a = torch.tensor([[1,2,],
                  [3,4]])
print(a.reshape(-1))
print(a.reshape(-1).reshape(2,2))