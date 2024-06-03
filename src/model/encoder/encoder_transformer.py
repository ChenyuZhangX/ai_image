import torch
import torch.nn as nn
from einops import rearrange
from model.transformer.transformer import Transformer

class Encoder_Transformer(nn.Module):
    def __init__(self, encoder_cfg):
        super(Encoder_Transformer, self).__init__()
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
        # x [latent_shape, latent_sensor]

        latent_shape = rearrange(x[0], "b (num h) -> b num h", h = self.d_latent)
        latent_sensor = rearrange(x[1], "b (num h) -> b num h", h = self.d_latent)

        x = torch.cat([latent_shape, latent_sensor], dim=1)
        x = self.transformer(x)
        x = rearrange(x, "b num h -> b (num h)")
        x = self.fc_out(x)
        return x
    
if __name__ == "__main__":
    # test the model
    encoder_cfg ={ # bs, 25 + 24 + 18, 24
        'd_in': [25, 24, 18],
        'd_latent': 24,
        'd_out': 25 * 24,
        'num_layers': 4,
        'transformer_cfg': {
            'dim': 24,
            'depth': 3,
            'heads': 8,
            'dim_head': 24,
            'mlp_dim': 256,
        }
    }   
    encoder = Encoder_Transformer(encoder_cfg)

    latent_sensor = [
        torch.randn(2, 24, 16),
        torch.randn(2, 18, 16),
        torch.randn(2, 24 + 18, 8),
    ]

    intra = torch.cat(latent_sensor[0:2], dim=1)
    cross = latent_sensor[-1]
    latent_sensor = torch.cat([intra, cross], dim = -1)

    print(latent_sensor.shape)

    latent_sensor = rearrange(latent_sensor, "b num h -> b (num h)")
    latent_shape = torch.randn(2, 25 * 24)

    out = encoder([latent_shape, latent_sensor])
    out = rearrange(out, "b (num h) -> b num h", h = 24)
    print(out.shape)