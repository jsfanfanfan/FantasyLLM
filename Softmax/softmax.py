import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x):
    x = torch.exp(x)
    x = x / x.sum()
    return x


x = torch.tensor([3.0, 2.0, 5.0, 1.0])
print(softmax(x))
# tensor([0.1125, 0.0414, 0.8310, 0.0152])