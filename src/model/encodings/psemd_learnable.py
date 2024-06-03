import torch
import torch.nn as nn


class PosEmbed_Learnable(nn.Module):
    def __init__(self, posembed_cfg):
        super(PosEmbed_Learnable, self).__init__()
        self.posembed_cfg = posembed_cfg
        self.posembed_num = posembed_cfg['posembed_num']
        self.posembed_dim_in = posembed_cfg['dim_in']
        self.posembed_dim = posembed_cfg['dim_out']
        
        self.embedding = []
        for idx in range(self.posembed_num):
            self.embedding.append(nn.Sequential(
                nn.Linear(self.posembed_dim_in[idx], self.posembed_dim[idx]),
                nn.ReLU(),
                nn.Linear(self.posembed_dim[idx], self.posembed_dim[idx]),
                nn.ReLU()
            ))
        self.embedding = nn.ModuleList(self.embedding)

    def forward(self, x: list):
        out = []
        for idx, xi in enumerate(x):
            out.append(self.embedding[idx](xi))
        return out
    
if __name__ == "__main__":
    posembed_cfg = {
        'posembed_num': 3,
        'dim_in': [10, 20, 30],
        'dim_out': [100, 200, 300],
    }
    posembed = PosEmbed_Learnable(posembed_cfg)
    x = [torch.randn(10, 10), torch.randn(10, 20), torch.randn(10, 30)]
    out = posembed(x)
    print(len(out))
    print("PosEmbed_Learnable test passed!")

