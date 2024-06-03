import torch
import torch.nn as nn
from einops import rearrange
from model.transformer.transformer import Transformer


class Decoder_Transformer(nn.Module):
    def __init__(self, decoder_cfg):
        super(Decoder_Transformer, self).__init__()
        self.cfg = decoder_cfg
        self.d_in = self.cfg['d_in']
        self.d_out = self.cfg['d_out']
        self.use_transformer = self.cfg.get('use_transformer', False)
        self.dim = self.cfg['transformer_cfg']['dim']

        if self.use_transformer:
            assert self.d_in % self.dim == 0, "d_in must be divisible by dim"
            self.transformer = Transformer(self.cfg['transformer_cfg'])

        self.out_fc = nn.Sequential()
        d_latent = self.d_in

        self.deeper = self.cfg.get('deeper', False)
        if self.deeper:
            while d_latent > 30 * self.d_out :
                self.out_fc.add_module(
                    f'fc_{d_latent}',
                    nn.Linear(d_latent, d_latent // 4),
                )
                self.out_fc.add_module(
                    f'relu_{d_latent}',
                    nn.ReLU(),
                )
                d_latent = d_latent // 4

            self.out_fc.add_module(
                'fc_out',
                nn.Linear(d_latent, self.d_out),
            )

            # softmax
            self.out_fc.add_module(
                'softmax_out',
                nn.Softmax(dim=1),
            )

        else:
            self.out_fc.add_module(
                'fc_out',
                nn.Linear(d_latent, self.d_out),
            )
    
    def forward(self, x):
        if self.use_transformer:
            x = rearrange(x, 'b (num d) -> b num d', d = self.dim)
            x = self.transformer(x)
            x = rearrange(x, 'b num d -> b (num d)')

        x = self.out_fc(x)
        return x

if __name__ == "__main__":
    decoder_cfg = {
        "d_in": 1024, 
        "d_out": 6, 
        "use_transformer": False,
        "transformer_cfg": {
            "dim": 1, 
            "depth": 3, 
            "heads": 20, 
            "dim_head": 20, 
            "mlp_dim": 128
        }

    }
    decoder = Decoder_Transformer(decoder_cfg)
    x = torch.randn(10, 1024)
    output = decoder(x)
    print(output.shape)