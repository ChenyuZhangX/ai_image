# MIT License

# Copyright (c) 2022 Karl Stelzner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file comes from https://github.com/stelzner/srt

from torch import nn

from .attention import Attention
from .feed_forward import FeedForward
from .pre_norm import PreNorm


class Transformer(nn.Module):
    def __init__(
        self,
        transformer_cfg,
    ):
        super().__init__()
        self.cfg = transformer_cfg
        self.layers = nn.ModuleList([])
        feed_forward_layer = self.cfg.get('feed_forward_layer', FeedForward)
        for _ in range(self.cfg['depth']):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            self.cfg['dim'],
                            Attention(
                                self.cfg['dim'],
                                heads=self.cfg['heads'],
                                dim_head=self.cfg['dim_head'],
                                dropout=self.cfg.get('dropout', 0.0),
                                selfatt=self.cfg.get('selfatt', True),
                                kv_dim=self.cfg.get('kv_dim', None),
                            ),
                        ),
                        PreNorm(self.cfg['dim'], 
                                feed_forward_layer(self.cfg['dim'], self.cfg['mlp_dim'], dropout=self.cfg.get('dropout', 0.0))),
                    ]
                )
            )

    def forward(self, x, z=None, **kwargs):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x, **kwargs) + x
        return x
