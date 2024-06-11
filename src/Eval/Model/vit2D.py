import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys 
sys.path.append(".../src/Eval")
sys.path.append(".../src/Eval/Model")
sys.path.append(".../src/Eval/Loss")
from position_encoding import PositionEmbeddingLearned2d


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return {
            "out": self.to_out(out),
            "attn": attn,  # batch heads tokens tokens
        }
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        attenMat = None
        for attn, ff in self.layers:
            x_out, attn_out = attn(x)["out"], attn(x)["attn"] 
            x = x_out + x
            x = ff(x) + x
            attenMat = attn_out
        return {
            "x": x,
            'attn': attenMat,
        }
    
class vit(nn.Module):
    def __init__(self, *, image_size, image_patch_size, dim, depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        # self.pool = pool
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_height = patch_height
        self.patch_width = patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    
        patch_dim = channels * patch_height * patch_width 
        
        self.to_patch_embedding_2D = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionEmbeddingLearned2d(dim // 2,(image_height // patch_height),(image_width // patch_width))
        self.dropout = nn.Dropout(emb_dropout)
        self.init_parameters()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def init_parameters(self):
        for m in self.to_patch_embedding_2D:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, video):
        B, C, H, W = video.shape
        x = self.to_patch_embedding_2D(video)

        pos = self.pos_embedding(H // self.patch_height, W // self.patch_width, x).unsqueeze(0).repeat(B, 1, 1)
        x += pos
        if self.pool == 'cls':
            x = torch.concat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.dropout(x)
        x_out = self.transformer(x)
        x, attn = x_out["x"], x_out["attn"]

        return x, pos, attn
    
class vitEmbed(nn.Module):
    def __init__(self, image_size=256, vis_dim=2048, patch_size=8, depth=12, heads=8, set_channel=1, pool='cls'):
        super().__init__()
        self.channel = set_channel
        self.pool = pool
        self.vision_encoder = vit(
            image_size = image_size,          # image size
            image_patch_size = patch_size,     # image patch size
            dim = vis_dim,
            depth = depth,
            heads = heads,
            mlp_dim = 2048,
            pool = pool,
            channels=set_channel,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        # self.embedding = nn.Embedding(set_channel,set_channel)
        self.mlp_embed= nn.Sequential(
            nn.Linear(vis_dim, vis_dim),
            nn.GELU(),
            nn.Linear(vis_dim, vis_dim)
        )
        self.init_parameters()
        self.vis_dim = vis_dim
        
    def init_parameters(self):
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x):
        x, pos_embedding, attn = self.vision_encoder(x) #x: B S D
        if self.pool != 'cls':
            x = x.mean(dim=1)
        else:
            x = x[:,0,:]
        x = self.mlp_embed(x)
        
        return x