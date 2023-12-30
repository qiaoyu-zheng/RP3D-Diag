import torch.nn as nn
import torch.nn.functional as F
import torch 
from einops import rearrange
import sys
sys.path.append("Path/to/RP3D-Diag")
sys.path.append("Path/to/RP3D-Diag/src")
from vit import vit
from einops.layers.torch import Rearrange
import random
from transformers import AutoTokenizer, AutoModel

class vitEmbed(nn.Module):
    def __init__(self, num_cls=917, vis_dim=768, frames=64, patch_size=32, frame_patch_size=4 , set_channel=3):
        super().__init__()
        self.channel = set_channel
        self.cls = num_cls
        self.vision_encoder = vit(
            image_size = 512,          # image size
            frames = frames,               # max number of frames
            image_patch_size = patch_size,     # image patch size
            frame_patch_size = frame_patch_size,      # frame patch size
            dim = vis_dim,
            depth = 12,
            heads = 8,
            mlp_dim = 2048,
            channels=set_channel,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        # self.embedding = nn.Embedding(set_channel,set_channel)
        self.mlp_embed= nn.Sequential(
            nn.Linear(vis_dim, vis_dim),
            nn.GELU(),
            nn.Linear(vis_dim, self.cls)
        )
        self.init_parameters()
        self.vis_dim = vis_dim
        
    def init_parameters(self):
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x, mod):
        x, pos_embedding = self.vision_encoder(x, mod) #x: B S D
        x = x.mean(dim=1)
        x = self.mlp_embed(x)
        
        return x


# class VIT2D_Embedding(nn.Module):
#     def __init__(self, vis_dim=768, depth=64, out_dim=1024, patch_size=32, frame_patch_size=4 , set_channel=3):
#         super().__init__()
#         self.channel = set_channel
#         self.vision_encoder = vit(
#             image_size = 512,          # image size
#             frames = depth,  
#             image_patch_size = patch_size,     # image patch size
#             frame_patch_size = frame_patch_size,      # frame patch size
#             dim = vis_dim,
#             depth = 12,
#             heads = 8,
#             mlp_dim = 2048,
#             dropout = 0.1,
#             emb_dropout = 0.1
#         )
#         # self.embedding = nn.Embedding(set_channel,set_channel)
#         self.mlp_embed= nn.Sequential(
#             nn.Linear(vis_dim, vis_dim),
#             nn.GELU(),
#             nn.Linear(vis_dim, out_dim)
#         )
#         self.init_parameters()
#         self.vis_dim = vis_dim
        
#     def init_parameters(self):
#         for m in self.mlp_embed:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal(m.weight, mode='fan_out')

#     def forward(self, x):
#         B,C,H,W = x.shape
#         # x = x.view(-1,1)
#         # x = self.embedding(x)
#         # x = x.view(B,H,W,D,self.channel)
#         x, pos_embedding = self.vision_encoder(x) #x: B S D
#         x = x.mean(dim=1)
#         x = self.mlp_embed(x)
        
#         return x

# class VIT3D_Embedding(nn.Module):
#     def __init__(self, vis_dim=768, depth=64, out_dim=1024, patch_size=32, frame_patch_size=4 , set_channel=3):
#         super().__init__()
#         self.channel = set_channel
#         self.vision_encoder = vit(
#             image_size = 512,          # image size
#             frames = depth,               # max number of frames
#             image_patch_size = patch_size,     # image patch size
#             frame_patch_size = frame_patch_size,      # frame patch size
#             dim = vis_dim,
#             depth = 12,
#             heads = 8,
#             mlp_dim = 2048,
#             dropout = 0.1,
#             emb_dropout = 0.1
#         )
#         # self.embedding = nn.Embedding(set_channel,set_channel)
#         self.mlp_embed= nn.Sequential(
#             nn.Linear(vis_dim, vis_dim),
#             nn.GELU(),
#             nn.Linear(vis_dim, out_dim)
#         )
#         self.init_parameters()
#         self.vis_dim = vis_dim
        
#     def init_parameters(self):
#         for m in self.mlp_embed:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal(m.weight, mode='fan_out')

#     def forward(self, x):
#         B,C,H,W,D = x.shape
#         # x = x.view(-1,1)
#         # x = self.embedding(x)
#         # x = x.view(B,H,W,D,self.channel)
#         x, pos_embedding = self.vision_encoder(x) #x: B S D
#         x = x.mean(dim=1)
#         x = self.mlp_embed(x)
        
#         return x
    
