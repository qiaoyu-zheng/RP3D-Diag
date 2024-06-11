import numpy as np
import sys
sys.path.append(".../src/Finetune")
sys.path.append(".../src/Finetune/Model")
sys.path.append(".../src/Finetune/Loss")
sys.path.append(".../src/Finetune/Utils")
from einops import rearrange
from resnet2D import resnet50_2D, ResNet50_Weights, resnet34_2D, ResNet34_Weights
from Loss.AllLosses import  MultiLabelLoss, MSEwithGaussian, KLwithGaussian, SoftCrossEntropy, InfoNCELoss
from vitFuse import vitFuse
from resAttention import Attention1D
import torch
from torch import nn
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, Resize, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandAdjustContrast
from safetensors.torch import load_model
from transformers import AutoModel,BertConfig,AutoTokenizer
from Utils.utils import visual_augment


rand_Gaussian_3d = RandGaussianNoise(
    prob=0.15,
    mean=0,
    std=0.07
)
rand_contrast_3d = RandAdjustContrast(
    prob=0.15,
    gamma=(0.5,1.5)
)

rand_affine = RandAffine(
    prob=0.15,
    rotate_range=(0, 0, np.pi/12),  
    translate_range=(10, 10, 0),  
    scale_range=(0.1, 0.1, 0),  
    shear_range=(0.2, 0.2, 0),  
    mode='bilinear',  
    padding_mode='zeros'  
)

rand_3d_elastic = Rand3DElastic(
    prob=0.15,  
    sigma_range=(5,7),
    magnitude_range=(10, 20),  
    rotate_range=(0, 0, 0), 
    scale_range=(0.1, 0.1, 0.1), 
    mode='bilinear',  
    padding_mode='border' 
)

transform = Compose([
    rand_affine,
    rand_3d_elastic,
    rand_Gaussian_3d,
    rand_contrast_3d,
])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class RadNet(nn.Module):
    def __init__(self, num_cls=5569, backbone='resnet', size=256, depth=32, hid_dim=2048, ltype='MultiLabel', augment=False, fuse='late', ke=False, encoded=None, adapter=False):
        super(RadNet, self).__init__()
        self.size = size
        self.depth = depth
        self.hid_dim = hid_dim
        self.pos_weights = torch.ones([num_cls]) * 20 
        self.cls = num_cls
        self.backbone = backbone
        self.fuse = fuse
        self.ke = True if ke == "True" else False
        self.adapter = True if adapter == "True" else False
        if encoded is not None:
            self.encoded = encoded
        if self.backbone == 'resnet':
            self.resnet2D = resnet50_2D(weights=("pretrained", ResNet50_Weights.DEFAULT))
            # self.resnet2D = resnet34_2D(weights=("pretrained", ResNet34_Weights.DEFAULT))
            self.attention1D = Attention1D(hid_dim=hid_dim, max_depth=depth, nhead=1, num_layers=1, pool='cls', batch_first=True)

        if fuse == 'early':
            self.vitFuse = vitFuse(mod_num=16, hid_dim=2048, nhead=4, num_layers=2, pool='cls', batch_first=True)

        self.fc2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, self.cls)
        )
        
        self.augment = augment
        self.ltype = ltype
        self.lossfunc = None
        if self.ltype == 'MultiLabel':
            self.lossfunc = MultiLabelLoss()
        else:
            raise ValueError("Invalid Loss Function")
        
        # self.attnloss = MSEwithGaussian(length=depth, std_dev_fraction=1/16)
        self.attnloss = KLwithGaussian(length=depth, std_dev_fraction=1/16)
        # self.attnloss = SoftCrossEntropy(length=depth, std_dev_fraction=1/16)
        # self.attnloss = InfoNCELoss(length=depth, temperature=0.07, thd=5)
        if self.ke:
            print("KEKE!")
            self.bioLORD = AutoModel.from_pretrained('FremyCompany/BioLORD-2023')
            for param in self.bioLORD.parameters():
                param.requires_grad = False
            self.fcke = nn.Linear(768, hid_dim)


    def forward(self, image_x, mods, marks, keys, labels):
        tmp_Aug = False
        labels = labels.squeeze(-1)
        keys = keys.squeeze(-1)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
        if self.training and self.augment:
            tmp_Aug = True
        if self.fuse != 'early':
            device = image_x.device
            B = image_x.shape[0]
            image_x = image_x.repeat(1,3,1,1,1)

            image_x = rearrange(image_x, 'b c h w d -> (b d) c h w')
            
            if tmp_Aug:
                image_x = rearrange(image_x, "b c h w -> b h w c")
                image_x = transform(image_x)
                # visual_augment(B1, image_x1, ".../src/Finetune/Logits/Aug_2D_2")
                image_x = rearrange(image_x, "b h w c -> b c h w")
            if self.backbone == 'resnet':
                output = self.resnet2D(image_x)
                output_x = output["x"]
                
            output_x = rearrange(output_x, '(b d) h -> b d h', b=B)
            

            output_x, attn_x, score_x = self.attention1D(output_x, marks)
            score_x = score_x[:,1:]
            
            cutoffs = marks.sum(dim=1).to(torch.long)
            loss_attn = self.attnloss(score_x, keys, cutoffs)

            if self.ke:
                self.encoded.to(device)
                ke_output = self.bioLORD(**self.encoded)
                embeds = mean_pooling(ke_output, self.encoded['attention_mask'])
                # Normalize embeddings
                embeds = F.normalize(embeds, p=2, dim=1)
                embeds = self.fcke(embeds)
                output_x = output_x @ embeds.T

            else:
                output_x = self.fc2(output_x)

            loss_cls = self.lossfunc(output_x, labels)
   
            return {
                'loss': loss_cls,
                'loss_return': loss_cls,
                'loss_attn': loss_cls,
                'loss_cls': loss_cls,
                'logits': output_x,
                'labels': labels,
            }

        else:
            device = image_x.device
            bs = image_x.shape[0]
            n = image_x.shape[1]
            Bs = bs * n
            image_x = image_x.repeat(1,1,3,1,1,1)
            image_x = rearrange(image_x, 'b n c h w d -> (b n d) c h w')
            marks = rearrange(marks, 'b n d -> (b n) d')
            mods = torch.squeeze(mods, dim=-1)
            keys = torch.squeeze(keys, dim=-1)
            if len(mods.shape) == 1:
                mods = mods.unsqueeze(0)
            if tmp_Aug:
                image_x = rearrange(image_x, "b c h w -> b h w c")
                image_x = transform(image_x)
                # visual_augment(B1, image_x1, ".../src/Finetune/Logits/Aug_2D_2")
                image_x = rearrange(image_x, "b h w c -> b c h w")
            if self.backbone == 'resnet':
                output = self.resnet2D(image_x)
                output_x= output["x"]

            output_x = rearrange(output_x, '(B d) h -> B d h', B=Bs)
            output_x, attn_x, score_x = self.attention1D(output_x, marks)

            score_x = score_x[:,1:]

            index = rearrange(mods, 'b n -> (b n)', b=bs)
            index_mask = (index!=0)
            score_x = score_x[index_mask]
            index_mark = marks[index_mask]
            cutoffs = index_mark.sum(dim=1).to(torch.long)
            keys = rearrange(keys, 'b n -> (b n)', b=bs)
            keys = keys[index_mask]
            loss_attn = self.attnloss(score_x, keys, cutoffs)

            output_x = rearrange(output_x, '(b n) h -> b n h', b=bs, n=n)
            output_x = self.vitFuse(output_x, mods)

            if self.ke:
                self.encoded.to(device)
                ke_output = self.bioLORD(**self.encoded)
                embeds = mean_pooling(ke_output, self.encoded['attention_mask'])
                # Normalize embeddings
                embeds = F.normalize(embeds, p=2, dim=1)
                embeds = self.fcke(embeds)
                output_x = output_x @ embeds.T
            else:
                output_x = self.fc2(output_x)
            loss_cls = self.lossfunc(output_x, labels)
            loss = loss_attn * 0.001 + loss_cls
            # loss_return = loss
            return {
                'loss': loss,
                'loss_return': loss,
                'loss_attn': loss_attn * 0.001,
                'loss_cls': loss_cls,
                'logits': output_x,
                'labels': labels,
            }

            
    



            
    

