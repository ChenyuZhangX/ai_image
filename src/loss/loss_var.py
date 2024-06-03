import torch
import torch.nn.functional as F
from torch import nn


class LossVAR(torch.nn.Module):
    def __init__(self):
        super(LossVAR, self).__init__()

    def __call__(self, x, y):
        # use torch var
        dev = x - y
        return torch.var(dev)
    
    def forward(self, x, y):
        return self.__call__(x, y)
    
if __name__ == "__main__":
    loss = LossVAR()
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    x = nn.Linear(10, 10)(x)
    losses = nn.ModuleList([loss])
    loss = 0.0
    for loss_fn in losses:
        loss += loss_fn(x, y)
    print(loss)
    loss.backward()
    
    print("LossVAR test passed!")