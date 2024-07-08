

import torch

# a = torch.tril(torch.ones(3,3), diagonal=0)
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.tensor([[2,7], [6,4],[6,5]], dtype=torch.float32)

# c = a @ b
# print(a)
# print(b)
# print(c)

a = torch.tensor(torch.arange(8)).repeat(4,1)
print(a)