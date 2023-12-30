import numpy as np
import sys
from einops import rearrange
sys.path.append("Path/to/RP3D-Diag")
sys.path.append("Path/to/RP3D-Diag/src")
sys.path.append("Path/to/RP3D-Diag/src/Model")
from resnet2D import resnet18_2D, ResNet18_Weights
from resnet3D import resnet18_3D
from resnetFuse import resnet34_Fuse
from vit_embedding import vitEmbed
from vitFuse import vitFuse
from medcpt import MedCPT_clinical
from Loss.AllLosses import MultiLabelLoss, BCEFocalLoss, MultiSoftLoss, AsymmetricLoss, ClipLoss
import torch
from torch import nn
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, Resize, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandAdjustContrast
from safetensors.torch import load_model
from transformers import AutoModel,BertConfig,AutoTokenizer

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


class RadNet(nn.Module):
    def __init__(self, num_cls=5569, backbone='resnet', depth=16, ltype='MultiLabel', augment=False, fuse='late', ke=False, encoded=None, adapter=False):
        super(RadNet, self).__init__()
        self.pos_weights = torch.ones([num_cls]) * 20 
        self.cls = num_cls
        self.backbone = backbone
        self.fuse = fuse
        self.ke = True if ke == "True" else False
        self.adapter = True if adapter == "True" else False
        if encoded is not None:
            self.encoded = encoded
        if self.backbone == 'resnet':
            self.resnet2D = resnet18_2D(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
            self.resnet3D = resnet18_3D(depth=depth)
            self.resnetFuse = resnet34_Fuse(num_classes=self.cls, fuse=self.fuse, ke=self.ke)
            
        else:
            self.vitEmbed = vitEmbed(num_cls=self.cls, frames=depth)
        
        if fuse == 'early':
            self.vitFuse = vitFuse(hid_dim=1024, num_cls=self.cls, ke=self.ke)
        
        self.augment = augment
        self.ltype = ltype
        self.lossfunc = None
        if self.ltype == 'MultiSoft':
            self.lossfunc = MultiSoftLoss()
        elif self.ltype == 'MultiLabel':
            self.lossfunc = MultiLabelLoss()
        elif self.ltype == 'BCEFocal':
            self.lossfunc = BCEFocalLoss(num_cls=self.cls, pos_weight=1, gamma=3.0, alpha=0.7, reduction='mean')
        elif self.ltype == 'Asymmetric':
            self.lossfunc = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True)
        elif self.ltype == 'Clip':
            self.lossfunc = ClipLoss()
        else:
            raise ValueError("Invalid Loss Function")

        if ke == "True":
            self.medcpt = MedCPT_clinical(bert_model_name='/remote-home/share/data200/172.16.11.200/zhengqiaoyu/pretrained')
            checkpoint = torch.load('/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/epoch_state2.pt',map_location='cpu')['state_dict']
            load_checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
            missing, unexpect = self.medcpt.load_state_dict(load_checkpoint, strict=False)
            print("missing_cpt:", missing)
            print("unexpect_cpt:", unexpect)
            for param in self.medcpt.parameters():
                param.requires_grad = False
            if adapter == "True":
                print("Adapter:", self.adapter)
                self.adaptFC = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )


    def forward(self, image_x, mods, dims, labels):
        tmp_Aug = False
        if self.training and self.augment:
            tmp_Aug = True
        if self.fuse != 'early':
            device = image_x.device
            B = image_x.shape[0]
            image_x = image_x.repeat(1,3,1,1,1)
            dims = torch.squeeze(dims)
            mask2D = (dims == 1)
            mask3D = (dims == 2)
            image_x1 = image_x[mask2D]
            image_x2 = image_x[mask3D]
            if self.backbone == 'resnet':
                output_x = torch.zeros((B,512,16,16),dtype=torch.float32,device=device)
            else:
                output_x = torch.zeros((B,self.cls),dtype=torch.float32,device=device)
            if image_x1.shape[0] > 0:
                B1 = image_x1.shape[0]
                image_x1 = image_x1[:,:,:,:,0]
                if tmp_Aug:
                    image_x1 = rearrange(image_x1, "b c h w -> b h w c")
                    image_x1 = transform(image_x1)
                    image_x1 = rearrange(image_x1, "b h w c -> b c h w")
                if self.backbone == 'resnet':
                    output1 = self.resnet2D(image_x1)
                    output_x1, output_C1, output_H1, output_W1 = output1["x"], output1["C"], output1["H"], output1["W"]
                    output_x[mask2D] = output_x1
                elif self.backbone == 'vit':
                    output_x1 = self.vitEmbed(image_x1, '2D')
                    output_x[mask2D] = output_x1
                else:
                    raise ValueError("Invalid Backbone in RadNet!")
                

            if image_x2.shape[0] > 0:
                B2 = image_x2.shape[0]
                if tmp_Aug:
                    image_x2 = image_x2[:,0,:,:,:]
                    image_x2 = transform(image_x2)
                    image_x2 = image_x2.unsqueeze(1).repeat(1,3,1,1,1)   
                if self.backbone == 'resnet':
                    output2 = self.resnet3D(image_x2)
                    output_x2, output_C2, output_H2, output_W2 = output2["x"], output2["C"], output2["H"], output2["W"]
                    output_x[mask3D] = output_x2
                elif self.backbone == 'vit':
                    output_x2 = self.vitEmbed(image_x2, '3D')
                    output_x[mask3D] = output_x2
                else:
                    raise ValueError("Invalid Backbone in RadNet!")
            
            if self.backbone == 'resnet':
                output_x = self.resnetFuse(output_x)
            elif self.backbone == 'vit':
                pass
            else:
                raise ValueError("Invalid Backbone in RadNet!")
            
            if self.ke:
                self.encoded.to(device)
                embeds = self.medcpt.encode_text(self.encoded)
                if self.adapter:
                    embeds = self.adaptFC(embeds)
                output_x = output_x @ embeds.T
    
            loss = self.lossfunc(output_x, labels)
            loss_return = loss
            return {
                'loss': loss,
                'loss_return': loss_return,
                'logits': output_x,
                'labels': labels,
            }

        else:
            device = image_x.device
            bs = image_x.shape[0]
            Bs = bs * image_x.shape[1]
            image_x = image_x.repeat(1,1,3,1,1,1)
            image_x = rearrange(image_x, 'b n c h w d -> (b n) c h w d')
            dims = rearrange(dims, 'b n d -> (b n) d')
            dims = torch.squeeze(dims)
            mods = torch.squeeze(mods)
            mask2D = (dims == 1)
            mask3D = (dims == 2)
            image_x1 = image_x[mask2D]
            image_x2 = image_x[mask3D]
            if self.backbone == 'resnet':
                output_x = torch.zeros((Bs,1024),dtype=torch.float32,device=device)
            else:
                output_x = torch.zeros((Bs,self.cls),dtype=torch.float32,device=device)
            if image_x1.shape[0] > 0:
                B1 = image_x1.shape[0]
                image_x1 = image_x1[:,:,:,:,0]
                if tmp_Aug:
                    image_x1 = rearrange(image_x1, "b c h w -> b h w c")
                    image_x1 = transform(image_x1)
                    image_x1 = rearrange(image_x1, "b h w c -> b c h w")
                if self.backbone == 'resnet':
                    output1 = self.resnet2D(image_x1)
                    output_x1, output_C1, output_H1, output_W1 = output1["x"], output1["C"], output1["H"], output1["W"]
                    output_x1 = self.resnetFuse(output_x1)
                    output_x[mask2D] = output_x1
                elif self.backbone == 'vit':
                    output_x1 = self.vitEmbed(image_x1, '2D')
                    output_x[mask2D] = output_x1
                else:
                    raise ValueError("Invalid Backbone in RadNet!")
                

            if image_x2.shape[0] > 0:
                B2 = image_x2.shape[0]
                if tmp_Aug:
                    image_x2 = image_x2[:,0,:,:,:]
                    image_x2 = transform(image_x2)
                    image_x2 = image_x2.unsqueeze(1).repeat(1,3,1,1,1)   
                if self.backbone == 'resnet':
                    output2 = self.resnet3D(image_x2)
                    output_x2, output_C2, output_H2, output_W2 = output2["x"], output2["C"], output2["H"], output2["W"]
                    output_x2 = self.resnetFuse(output_x2)
                    output_x[mask3D] = output_x2
                elif self.backbone == 'vit':
                    output_x2 = self.vitEmbed(image_x2, '3D')
                    output_x[mask3D] = output_x2
                else:
                    raise ValueError("Invalid Backbone in RadNet!")
            
            output_x = self.vitFuse(output_x, mods)

            if self.ke:
                self.encoded.to(device)
                embeds = self.medcpt.encode_text(self.encoded)
                if self.adapter:
                    embeds = self.adaptFC(embeds)
                output_x = output_x @ embeds.T

            loss = self.lossfunc(output_x, labels)
            loss_return = loss
            return {
                'loss': loss,
                'loss_return': loss_return,
                'logits': output_x,
                'labels': labels,
            }

            
    



            
    

