import math
import torch
from torch import nn
from einops import rearrange, repeat


class PositionEmbeddingLearned2d(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256,h_patch_num = 8, w_patch_num = 8):
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, h, w, x):
        i = (torch.arange(h, device=x.device) + 1) - 1
        j = (torch.arange(w, device=x.device) + 1) - 1
        x_emb = self.row_embed(i).unsqueeze(1).repeat(1,w,1)
        y_emb = self.col_embed(j).unsqueeze(0).repeat(h,1,1)
        pos = torch.cat([x_emb,y_emb], dim=-1)
        pos = rearrange(pos,'h w c -> (h w) c')
        return pos