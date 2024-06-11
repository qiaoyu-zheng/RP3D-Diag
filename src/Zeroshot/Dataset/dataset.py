import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoModel
import numpy as np
import random
import nibabel as nib
import scipy
import ruamel.yaml as yaml
from tqdm import tqdm
import sys
import random
from einops import rearrange
import os
import re
import pickle
from PIL import Image
random.seed(42)

Modality_dict = {"CT":[1,2,3], "MRI":[4,5,6], "X-ray": [7,8], "Ultrasound": [9], "Annotated image": [10], "Fluoroscopy": [11], 
                 "Mammography": [12], "DSA (angiography)": [13], "Nuclear medicine": [14], "Barium": [15]}

def resample_image_3D(image, tS, tD):
    height = image.shape[-3]
    width = image.shape[-2]
    assert height == width
    size = height
    depth = image.shape[-1]

    if size== tS and depth == tD: 
        output_tensor = image
    else:
        # Upsample or Downsample
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS, tD), mode='trilinear', align_corners=False)
    return output_tensor

def resample_image_2D(image, tS):
    height = image.shape[-2]
    width = image.shape[-1]
    assert height == width
    size = height

    if size == tS:
        output_tensor = image
    else:
        # Upsample or Downsample
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS), mode='bilinear', align_corners=False)
    return output_tensor

def resample_image(image, tS, tD):
    depth = image.shape[0]
    height = image.shape[-2]
    width = image.shape[-1]
    # assert height == width
    size = height

    if size == tS and width == tS and depth <= tD:
        output_tensor = image
    else:
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS), mode='bilinear', align_corners=False)
        if depth > tD:
            step = depth / tD
            indices = indices = torch.arange(0, depth, step).long()
            indices = indices[:tD]
            output_tensor = output_tensor[indices]
    return output_tensor

def select_images(folder_path, d):
    files = sorted(os.listdir(folder_path))
    
    total_files = len(files)
    if total_files == 0:
        raise ValueError("No png files found in the folder.")
    d = min(d, total_files)
    
    step = total_files // d
    selected_files = [files[i] for i in range(0, total_files, step)][:d]
    
    return selected_files

def load_images_as_grayscale(folder_path, file_list):
    images = []
    for file_name in file_list:
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path).convert('L') 
        img_array = np.array(img)
        images.append(img_array)
    
    stacked_images = np.stack(images, axis=-1)
    
    return stacked_images

class RadNet_Dataset_late(Dataset):
    def __init__(self, data_path, label_path, num_classes=5569, level='articles', size=256, depth=32):
        with open(data_path, 'r') as f:
            self.data_dict = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.level = level
        self.size = size
        self.depth = depth
        self.link_list = list(self.data_dict.keys())
        self.folder_dir = "/mnt/petrelfs/share_data/zhengqiaoyu.p/processed_index"

    
    def __getitem__(self, index):
        link = self.link_list[index]
        
        Modalities = self.data_dict[link]["Modalities"]
        Samples = self.data_dict[link]["Samples"]
        length = len(Modalities)
        randnum = random.randrange(0,length)
        mark = torch.zeros(self.depth)
        #process label
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        label_key_list = Samples[randnum][self.level]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        sample = Samples[randnum]
        randpick = random.randrange(0,len(sample["image_path"]))
        image_path = sample["image_data_path"][randpick]
        key_path = sample["image_key_path"][randpick]
        image_datas = np.load(image_path)
        if image_datas.shape[-1] == 3 and (image_datas[:,:,0] == image_datas[:,:,1]).all() and (image_datas[:,:,0] == image_datas[:,:,2]).all():
            image_datas = image_datas[:,:,:1]
        tensor_keys = torch.tensor(np.load(key_path), dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            print("warning: 0")
            print(link)
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_dict)
    
class ChestXDet10(Dataset):
    def __init__(self, data_path, label_path, num_classes=11, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
        
    
    def __getitem__(self, index):
        data = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = data["syms"]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = data["file_name"]
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    

class CheXpert(Dataset):
    def __init__(self, data_path, label_path, num_classes=14, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)

class COVID19_Rad(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    
class IU_Xray(Dataset):
    def __init__(self, data_path, label_path, num_classes=2, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    
class PadChest(Dataset):
    def __init__(self, data_path, label_path, num_classes=193, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    
class BrainTumor(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.datas)
    
class BrainTumor17(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    
class CT_Kidney(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.datas)
    
class KneeMRI(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        # loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[label_key_list] = 1
        labels = labels.to(torch.float32)
        #process image
        file_path = path
        with open(file_path, 'rb') as file:
            image_datas = pickle.load(file)
        image_datas = image_datas.astype(np.float32)
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0)
        image_tensor = rearrange(image_tensor, "c d h w -> c h w d")
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_list)
    
class MURA(Dataset):
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        try:
            path = self.datas[index]
            image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
            mark = torch.zeros(self.depth)
            label_key_list = self.data_list[path]
            label_key_list = list(set(label_key_list))
            labels = torch.zeros(self.cls)
            labels[label_key_list] = 1
            labels = labels.to(torch.float32)
            #process image
            image_path = path
            image_datas = np.array(Image.open(image_path).convert('L'))
            
            tensor_keys = torch.tensor([0], dtype=torch.long)
            modality = torch.tensor([0])
            image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            # print("image_tensor", image_tensor.shape)
            image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
            image_tensor = resample_image(image_tensor, self.size, self.depth)
            image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
            
            if tensor_keys[0] > image_tensor.shape[-1]:
                selected_index = random.randrange(0,image_tensor.shape[-1])
                tensor_keys[0] = selected_index
            mark[:image_tensor.shape[-1]] = 1
            if image_tensor.max()-image_tensor.min() == 0:
                image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
            image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
            image_fuse[:,:,:,0:image.shape[-1]] = image
            return {
                "images": image_fuse, # (1, 512, 512, depth)
                "mods": modality,
                "marks": mark, # (depth)
                "keys": tensor_keys,
                "labels": labels,
                }
        except:
            print(path)
            return {
                "images": image_fuse, # (1, 512, 512, depth)
                "mods": modality,
                "marks": mark, # (depth)
                "keys": tensor_keys,
                "labels": labels,
                }
        
    
    def __len__(self):
        return len(self.datas)

class POCUS(Dataset):
    def __init__(self, data_path, label_path, num_classes=3, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.datas)
    
class VinDr_Spine(Dataset):
    def __init__(self, data_path, label_path, num_classes=3, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_path = path
        image_datas = np.array(Image.open(image_path).convert('L'))
        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.datas)

    
class CC_CCII(Dataset):
    def __init__(self, data_path, label_path, num_classes=3, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index]
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)
        #process image
        image_ids = select_images(path, self.depth)  
        image_datas = load_images_as_grayscale(path, image_ids).astype(np.float32)

        
        tensor_keys = torch.tensor([0], dtype=torch.long)
        modality = torch.tensor([0])
        image_tensor = torch.tensor(image_datas, dtype=torch.float32).unsqueeze(0)
        # print("image_tensor", image_tensor.shape)
        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.size, self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        if tensor_keys[0] > image_tensor.shape[-1]:
            selected_index = random.randrange(0,image_tensor.shape[-1])
            tensor_keys[0] = selected_index
        mark[:len(image_ids)] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        return {
            "images": image_fuse, # (1, 512, 512, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
    
    def __len__(self):
        return len(self.datas)

