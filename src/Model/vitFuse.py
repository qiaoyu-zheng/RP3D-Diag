import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys 
sys.path.append("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision_Multi")
sys.path.append("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision_Multi/Model")
sys.path.append("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision_Multi/Loss")


class vitFuse(nn.Module):
    def __init__(self, hid_dim=1024, num_cls=917, ke=False):
        super().__init__()
        # self.mod_embed = nn.Embedding(n+1, hid_dim)
        self.mod_embed = nn.Parameter(torch.randn(15, hid_dim, dtype=torch.float32))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim,nhead=8,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim, dtype=torch.float32))
        self.cls = num_cls
        self.ke = ke
        if ke:
            self.fcke = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.GELU(),
                nn.Linear(hid_dim, 768)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.GELU(),
                nn.Linear(hid_dim, self.cls)
            )
        self.init_parameters()

    def init_parameters(self):
        if self.ke:
            for m in self.fcke:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
        else:
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x, mod):
        B, N = mod.shape
        mod = rearrange(mod, 'b n -> (b n)')
        modality_embedding = self.mod_embed[mod]
        x += modality_embedding
        x = rearrange(x, '(b n) d -> b n d', b=B)
        mod = rearrange(mod, '(b n) -> b n', b=B)
        mask = (mod==0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        mask = F.pad(mask, (1, 0), "constant", 0)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x[:,0,:]
        if self.ke:
            x = self.fcke(x)
        else:
            x = self.fc(x)
        return x

        


        

        
        


        

