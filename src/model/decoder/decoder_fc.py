import torch
import torch.nn as nn


class Decoder_FC(nn.Module):
    def __init__(self, decoder_cfg):
        super(Decoder_FC, self).__init__()
        self.cfg = decoder_cfg
        self.d_in = self.cfg['d_in']
        self.d_out = self.cfg['d_out']
        self.d_hidden = self.cfg['d_hidden']
        
        # Batch Norm will get (1, xxx) as 
        self.fc_in = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_hidden),
            nn.Linear(self.d_hidden, self.d_hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_hidden // 2),
        )

        self.out = nn.Sequential(
            nn.Linear(self.d_hidden // 2, self.d_out),
        )
        
        self.restriction = nn.Sequential(
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    decoder_cfg = {
        'd_in': 512,
        'd_out': 6,
        'd_hidden': 256
    }
    decoder = Decoder_FC(decoder_cfg)
    x = torch.randn(10, 512)
    output = decoder(x)
    print(output.shape)