import torch
import torch.nn as nn
from einops import rearrange
from model.transformer.transformer import Transformer

class Encoder_Shape(nn.Module):
    def __init__(self, encoder_cfg):
        super(Encoder_Shape, self).__init__()
        self.cfg = encoder_cfg
        self.d_in = self.cfg['d_in']
        self.d_latent = self.cfg['d_latent']
        self.d_out = self.cfg['d_out']
        self.num_layers = self.cfg['num_layers']
        transformer_cfg = self.cfg['transformer_cfg']
        self.transformer = nn.Sequential()

        for _ in range(self.num_layers):
            self.transformer.add_module('transformer', Transformer(transformer_cfg))
        
        self.fc_out = nn.Sequential(
            nn.Linear(self.d_latent * sum(self.d_in), self.d_out),
            nn.ReLU(),
            nn.Linear(self.d_out, self.d_out),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x embedding of the latent_shape
        latent = rearrange(x, "b (num h) -> b num h", h = self.d_latent)
        x = latent
        x = self.transformer(x)
        x = rearrange(x, "b num h -> b (num h)")
        x = self.fc_out(x)
        return x
    
if __name__ == "__main__":
    # test the model
    encoder_shape_cfg =  {
        'd_in': 9,
        'd_latent': 3 * 20,
        'd_out': 25 * 24,
        'num_layers': 4,
        'transformer_cfg': {
            'dim': 24,
            'depth': 3,
            'heads': 8,
            'dim_head': 24,
            'mlp_dim': 128,
        },
    }
   
    encoder = Encoder_Shape(encoder_shape_cfg)
    latent_shape = torch.randn(2, 9 * 60)

    out = encoder(latent_shape)
    out = rearrange(out, "b (num h) -> b num h", h = 24)
    print(out.shape)