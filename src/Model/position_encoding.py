import math
import torch
from torch import nn
from einops import rearrange, repeat

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256,h_patch_num = 16, w_patch_num = 16,d_patch_num = 64):
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, B, h, w, d, x, mod='3D'):
        i = (torch.arange(h, device=x.device) + 1) - 1
        j = (torch.arange(w, device=x.device) + 1) - 1
        k = (torch.arange(d, device=x.device) + 1) - 1
        x_emb = self.row_embed(i).unsqueeze(1).unsqueeze(2).repeat(1,w,d,1)
        y_emb = self.col_embed(j).unsqueeze(0).unsqueeze(2).repeat(h,1,d,1)
        z_emb = self.dep_embed(k).unsqueeze(0).unsqueeze(1).repeat(h,w,1,1)
        pos = torch.cat([x_emb,y_emb,z_emb,], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        if mod == '3D':
            pos = rearrange(pos, 'b h w d c -> b (h w d) c')
            return pos
        elif mod == '2D':
            pos = pos[:,:,:,0,:]
            pos = rearrange(pos, 'b h w c -> b (h w) c')
            return pos
        else:
            raise ValueError("Invalid Dimention in PositionEmbeddingLearned!")