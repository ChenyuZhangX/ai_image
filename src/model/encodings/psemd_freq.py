import torch
import torch.nn as nn


def embed_pos_freq(x, dim):
    n, d = x.shape
    freqs = torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
    freqs = 1 / torch.pow(10000, (freqs / d))
    pos = torch.arange(0, n, device=x.device, dtype=x.dtype).unsqueeze(1)
    pos = torch.matmul(pos, freqs.unsqueeze(0))
    pos = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
    return pos

class PosEmbed_Freq(nn.Module):
    def __init__(self, posembed_cfg):
        super(PosEmbed_Freq, self).__init__()
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

