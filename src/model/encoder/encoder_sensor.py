import torch
import torch.nn as nn
from einops import rearrange
from model.transformer.transformer import Transformer

class Encoder_Sensor(nn.Module):
    def __init__(self, encoder_cfg):
        super(Encoder_Sensor, self).__init__()
        self.cfg = encoder_cfg
        self.d_in = self.cfg['d_in']
        self.d_latent = self.cfg['d_latent']
        self.d_out = self.cfg['d_out']


        self.intra_top = self.cfg['intra_top']
        self.intra_bot = self.cfg['intra_bot']
        self.cross = self.cfg['cross']

        self.intra_top = Transformer(self.intra_top)
        self.intra_bot = Transformer(self.intra_bot)
        self.cross = Transformer(self.cross)
        
        self.fc_top = nn.Sequential(
            nn.Linear(self.d_latent * self.d_in[0], self.d_out[0] * self.d_in[0]),
            nn.ReLU(),
        )

        self.fc_bot = nn.Sequential(
            nn.Linear(self.d_latent * self.d_in[1], self.d_out[1] * self.d_in[1]),
            nn.ReLU(),
        )

        self.fc_cross = nn.Sequential(
            nn.Linear(self.d_latent * sum(self.d_in), self.d_out[2] * sum(self.d_in)),
            nn.ReLU(),
        )

    
    def forward(self, x):
        # embedding of sensor data [top, bot]
        top = x[0]
        bot = x[1]

        top = rearrange(top, "b (num h) -> b num h", h = self.d_latent)
        bot = rearrange(bot, "b (num h) -> b num h", h = self.d_latent)

        intra_top = self.intra_top(top).reshape(top.shape[0], -1)
        intra_top = self.fc_top(intra_top)

        intra_bot = self.intra_bot(bot).reshape(bot.shape[0], -1)
        intra_bot = self.fc_bot(intra_bot)

        cross = self.cross(torch.cat([top, bot], dim = 1)).reshape(top.shape[0], -1)
        cross = self.fc_cross(cross)

        intra_top = rearrange(intra_top, "b (num h) -> b num h", h = self.d_out[0])
        intra_bot = rearrange(intra_bot, "b (num h) -> b num h", h = self.d_out[1])
        cross = rearrange(cross, "b (num h) -> b num h", h = self.d_out[2])

        intra = torch.cat([intra_top, intra_bot], dim = 1)
        x = torch.cat([intra, cross], dim = -1)
        x = rearrange(x, "b num h -> b (num h)")

        return x
    
if __name__ == "__main__":
    # test the model
    encoder_sensor_cfg = {
        'd_in': [24, 18],
        'd_latent': 16,
        'd_out': [16, 16, 8],
        'intra_top': {
            'dim': 16,
            'depth': 3,
            'heads': 8,
            'dim_head': 24,
            'mlp_dim': 64,
        },
        'intra_bot': { 
            'dim': 16,
            'depth': 3,
            'heads': 8,
            'dim_head': 24,
            'mlp_dim': 64,
        },
        'cross': {
            'dim': 16,
            'depth': 3,
            'heads': 8,
            'dim_head': 24,
            'mlp_dim': 64,
        }
    }
    encoder = Encoder_Sensor(encoder_sensor_cfg)

    embed_sensor = [
        torch.randn(2, 24 * 16),
        torch.randn(2, 18 * 16),
    ]

    latent_sensor = encoder(embed_sensor)
    latent_sensor = rearrange(latent_sensor, "b (num h) -> b num h", h = 24)
    print(latent_sensor.shape)