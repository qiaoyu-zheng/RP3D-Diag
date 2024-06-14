import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys 

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys 
import copy
from typing import Optional, Any, Union, Callable
import torch
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np

from transformer_encoder import TransformerEncoder

class Attention1D(nn.Module):
    def __init__(self, hid_dim=2048, max_depth=32, nhead=8, num_layers=6, pool='cls', batch_first=True):
        super().__init__()
        
        self.transformer_encoder = TransformerEncoder(d_model=hid_dim, d_ff=hid_dim, n_heads=nhead, n_layers=num_layers, dropout=0.)

        self.pool = pool
        self.batch_first = batch_first
        if pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim, dtype=torch.float32))
            self.pos_emb = nn.Parameter(torch.randn(1+max_depth, hid_dim, dtype=torch.float32))
        else:
            self.pos_emb = nn.Parameter(torch.randn(max_depth, hid_dim, dtype=torch.float32))
        
        
    def forward(self, x, mask):
        B, S, D = x.shape
        # print("mask", mask)
        if self.pool == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  
            x = torch.cat((cls_tokens, x), dim=1)
            mask = F.pad(mask, (1, 0), "constant", 1)
        elif self.pool == 'mean':
            pass
        else:

            raise ValueError('pool type must be either cls (cls token) or mean (mean pooling)')
        
        x = x + self.pos_emb

        if not self.batch_first:
            x = x.permute(1, 0, 2)
            mask = mask.permute(1, 0)

        # print("mask", mask.to(torch.bool))
        x, attn_x, score_x = self.transformer_encoder(x, mask.to(torch.bool))

        attn_x = attn_x.mean(dim=1)
        attn_x = attn_x[:,0,:]
        score_x = score_x.mean(dim=1)
        score_x = score_x[:,0,:]

        if not self.batch_first:
            x = x.permute(1, 0, 2)

        if self.pool == 'cls':
            x = x[:,0,:]
        else:
            x = x.mean(dim=1)
        return x, attn_x, score_x
    
if __name__ == "__main__":
    transformer_encoder = TransformerEncoder(d_model=768, d_ff=2048, n_heads=1, n_layers=1, dropout=0.)
    x = torch.randn((1,4,768), dtype=torch.float32)
    mask = torch.randint(0,2,(1,4), dtype=torch.float32)
    print("mask", mask)
    print(x.shape)
    print(mask.shape)
    x, attn_x = transformer_encoder(x, mask.to(torch.bool))
    print(attn_x)
    print(attn_x.shape)
    print(x)
    # attn1D = Attention1D(hid_dim=768, max_depth=4, nhead=1, num_layers=1, pool='cls', batch_first=True)