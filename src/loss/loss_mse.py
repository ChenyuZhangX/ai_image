import torch
import torch.nn.functional as F
from torch import nn


class LossMSE(torch.nn.Module):
    def __init__(self):
        super(LossMSE, self).__init__()

    def __call__(self, x, y):
        # use torch mse
        return F.mse_loss(x, y)
    
    def forward(self, x, y):
        return self.__call__(x, y)
    
if __name__ == "__main__":
    loss = LossMSE()
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    x = nn.Linear(10, 10)(x)
    losses = nn.ModuleList([loss])
    loss = 0.0
    for loss_fn in losses:
        loss += loss_fn(x, y)

    loss.backward()
    print("LossMSE test passed!")