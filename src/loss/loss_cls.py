import torch
import torch.nn.functional as F
from torch import nn

# cross entropy loss
class LossCLS(torch.nn.Module):
    def __init__(self):
        super(LossCLS, self).__init__()

    def __call__(self, x, y):
        # use torch mse
        return F.cross_entropy(x, y)
    
    def forward(self, x, y):
        return self.__call__(x, y)
    
if __name__ == "__main__":
    loss_cls = LossCLS()
    x = torch.rand((10, 10))
    x = nn.Linear(10, 10)(x)
    # softmax
    x = F.softmax(x, dim=1)
    print(x.shape)
    y = torch.randint(0, 10, (10,))
    # y as one hot
    y = F.one_hot(y, num_classes=10)
    y = y.float()
    print(y.shape)

    losses = nn.ModuleList([loss_cls])

    loss = 0.0
    for loss_fn in losses:
        loss += loss_fn(x, y)

    loss.backward()

    print("LossCLS test passed!")

