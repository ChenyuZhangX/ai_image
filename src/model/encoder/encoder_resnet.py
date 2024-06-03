import torch
import torch.nn as nn

class Encoder_ResNet(nn.Module):

    def __init__(self, encoder_cfg):
        super(Encoder_ResNet, self).__init__()
        self.cfg = encoder_cfg
        self.d_in = self.cfg['d_in']
        self.d_out = self.cfg['d_out']
        self.d_hidden = self.cfg['d_hidden']
        self.n_resblk1 = self.cfg['n_resblk1']
        self.n_resblk2 = self.cfg['n_resblk2']
        
        self.fc_in = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_hidden // 2),
            nn.Linear(self.d_hidden // 2, self.d_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_hidden)
        )

        self.resblk1 = []
        for _ in range(self.n_resblk1):
            self.resblk1.append(nn.Sequential(
                nn.Linear(self.d_hidden + self.d_in, self.d_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_hidden),
                nn.Linear(self.d_hidden, self.d_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_hidden)
            ))
        self.resblk1 = nn.ModuleList(self.resblk1)

        self.resblk2 = []
        for _ in range(self.n_resblk2):
            self.resblk2.append(nn.Sequential(
                nn.Linear(self.d_hidden + self.d_hidden, self.d_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_hidden),
            ))
        self.resblk2 = nn.ModuleList(self.resblk2)
            
        self.fc_out = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_out)
        )

    def forward(self, x):
        input = x

        x = self.fc_in(x)
        # skip connection
        for resblk in self.resblk1:
            x = torch.cat([x, input], dim=-1)
            x = resblk(x)
        
        # skip connection
        x_prev = x
        for resblk in self.resblk2:
            x = torch.cat([x, x_prev], dim=-1)
            x = resblk(x)
        
        x = self.fc_out(x)
        return x
    
if __name__ == "__main__":
    # test the model
    encoder_cfg = {
        'd_in': 10,
        'd_out': 5,
        'd_hidden': 20,
        'n_resblk1': 2,
        'n_resblk2': 3
    }
    encoder = Encoder_ResNet(encoder_cfg)
    x = torch.randn(2, encoder_cfg['d_in'])
    y = encoder(x)
    print(y.shape)

    # print all the parameters
    for name, param in encoder.named_parameters():
        print(name, param.shape)