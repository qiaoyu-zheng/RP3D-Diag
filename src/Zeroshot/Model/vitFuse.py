import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys 
from transformer_encoder import TransformerEncoder
sys.path.append(".../src/Zeroshot")
sys.path.append(".../src/Zeroshot/Model")
sys.path.append(".../src/Zeroshot/Loss")


class vitFuse(nn.Module):
    def __init__(self, mod_num=16, hid_dim=2048, num_cls=917, nhead=1, num_layers=1, pool='cls', batch_first=True):
        super().__init__()
        # self.mod_embed = nn.Embedding(n+1, hid_dim)
        self.mod_embed = nn.Parameter(torch.randn(mod_num, hid_dim, dtype=torch.float32))
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim,nhead=nhead,batch_first=batch_first)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.transformer_encoder = TransformerEncoder(d_model=hid_dim, d_ff=hid_dim, n_heads=nhead, n_layers=num_layers, dropout=0.)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim, dtype=torch.float32))
        self.cls = num_cls
        # self.fc = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.GELU(),
        #     nn.Linear(hid_dim, self.cls)
        # )
        # self.init_parameters()

    # def init_parameters(self):
    #     for m in self.fc:
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x, mod):
        B, N = mod.shape
        mod = rearrange(mod, 'b n -> (b n)')
        modality_embedding = self.mod_embed[mod]
        modality_embedding = rearrange(modality_embedding, '(b n) d -> b n d', b=B)
        x += modality_embedding
        # x = rearrange(x, '(b n) d -> b n d', b=B)
        mod = rearrange(mod, '(b n) -> b n', b=B)
        mask = (mod!=0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        mask = F.pad(mask, (1, 0), "constant", 1)
        x, attn, score = self.transformer_encoder(x, mask.to(torch.bool))
        x = x[:,0,:]
        # x = self.fc(x)
        return x