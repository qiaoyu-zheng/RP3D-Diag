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
    assert height == width
    size = height

    if size == tS and depth <= tD:
        output_tensor = image
    else:
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS), mode='bilinear', align_corners=False)
        if depth > tD:
            step = depth / tD
            indices = indices = torch.arange(0, depth, step).long()
            indices = indices[:tD]
            output_tensor = output_tensor[indices]
    return output_tensor

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
        if "Dim" in self.data_dict[link] and self.data_dict[link]["Dim"] == "3D":
            try:
                assert tensor_keys[0] < image_datas.shape[-1] 
                image_datas = image_datas[:,:, tensor_keys[0]:tensor_keys[0]+1]
            except:
                image_datas = image_datas[:,:, image_datas.shape[-1]//2:image_datas.shape[-1]//2+1]
            tensor_keys[0] = 0

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
            "images": image_fuse, # (1, 256, 256, depth)
            "mods": modality,
            "marks": mark, # (depth)
            "keys": tensor_keys,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.data_dict)
    
class RadNet_Dataset_early(Dataset):
    def __init__(self, data_path, label_path, num_classes=5569, level='articles', size=256, depth=32, n_image=6):
        with open(data_path, 'r') as f:
            self.data_dict = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.level = level
        self.size = size
        self.depth = depth
        self.n_image = n_image
        self.link_list = list(self.data_dict.keys())
        self.folder_dir = "/mnt/petrelfs/share_data/zhengqiaoyu.p/processed_index"

    def __getitem__(self, index):
        link = self.link_list[index]
        Modalities = self.data_dict[link]["Modalities"]
        Samples = self.data_dict[link]["Samples"]

        # 将列表元素配对并打乱
        combined = list(zip(Modalities, Samples))
        random.shuffle(combined)

        # 重新分离这两个列表
        Modalities_shuffled, Samples_shuffled = zip(*combined)

        # 将它们从元组转换回列表（如果需要的话）
        Modalities = list(Modalities_shuffled)
        Samples = list(Samples_shuffled)

        length = len(Modalities)
        labels = None
        image_fuse = torch.zeros((self.n_image,1,self.size,self.size,self.depth), dtype=torch.float32)
        modality = torch.zeros((self.n_image,1), dtype=torch.long)
        mark = torch.zeros((self.n_image,self.depth), dtype=torch.long)
        key = torch.full((self.n_image,1), -1, dtype=torch.long)
        cnt = 0
        for i in range(length):
            if cnt >= self.n_image:
                break
            if labels is None:
                #process label
                label_key_list = Samples[i][self.level]
                label_key_list = list(set(label_key_list))
                loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
            
                labels = torch.zeros(self.cls)
                labels[loc_list] = 1
                labels = labels.to(torch.float32)

            if Modalities[i] == 'CT' or Modalities[i] == 'MRI':
                axis = len(Samples[i]["image_data_path"])
                axis = min(axis ,3)
                image_datas = np.zeros((axis, self.size, self.size, self.depth), dtype=np.float32)
                image_keys = np.full((axis, 1), -1, dtype=np.int32)
                image_marks = np.zeros((axis, self.depth), dtype=np.int32)
                image_modalitys = np.zeros((axis, 1), dtype=np.int32)
                for k in range(axis):
                    image_data = np.load(Samples[i]["image_data_path"][k])
                    image_key = np.load(Samples[i]["image_key_path"][k])
                    d = image_data.shape[-1]
                    if image_key[0] >= d:
                        selected_index = random.randrange(0,d)
                        image_key[0] = selected_index
                    image_datas[k,:,:,:d] = image_data
                    image_keys[k,0] = image_key[0]
                    length = min(d, self.depth)
                    image_marks[k,:length] = 1
                    image_modalitys[k,0] = Modality_dict[Modalities[i]][k]
 
            elif Modalities[i] == 'X-ray':
                axis = len(Samples[i]["image_data_path"])
                axis = min(axis, 2)
                image_datas = np.zeros((axis, self.size, self.size, self.depth), dtype=np.float32)
                image_keys = np.full((axis,1), -1, dtype=np.int32)
                image_marks = np.zeros((axis, self.depth), dtype=np.int32)
                image_modalitys = np.zeros((axis, 1), dtype=np.int32)
                for k in range(axis):
                    image_data = np.load(Samples[i]["image_data_path"][k])
                    image_key = np.load(Samples[i]["image_key_path"][k])
                    d = image_data.shape[-1]
                    if image_key[0] >= d:
                        selected_index = random.randrange(0,d)
                        image_key[0] = selected_index
                    image_datas[k,:,:,:d] = image_data
                    image_keys[k,0] = image_key[0]
                    length = min(d, self.depth)
                    image_marks[k,:length] = 1
                    image_modalitys[k,0] = Modality_dict[Modalities[i]][k]
 
            else:
                axis = len(Samples[i]["image_data_path"])
                image_datas = np.zeros((1, self.size, self.size, self.depth), dtype=np.float32)
                image_keys = np.full((1,1), -1, dtype=np.int32)
                image_marks = np.zeros((1, self.depth), dtype=np.int32)
                image_modalitys = np.zeros((1, 1), dtype=np.int32)
                rand_pick = random.randrange(0, axis)
                image_data = np.load(Samples[i]["image_data_path"][rand_pick])
                d = image_data.shape[-1]
                image_key = np.load(Samples[i]["image_key_path"][rand_pick])
                if image_key[0] >= d:
                    selected_index = random.randrange(0,d)
                    image_key[0] = selected_index
                image_datas[0,:,:,:d] = image_data
                image_keys[0,0] = image_key[0]
                length = min(d, self.depth)
                image_marks[0,:length] = 1
                image_modalitys[0,0] = Modality_dict[Modalities[i]][0]


            if cnt + image_datas.shape[0] > self.n_image:
                image_datas = image_datas[:self.n_image - cnt]
                image_keys = image_keys[:self.n_image - cnt]
                image_marks = image_marks[:self.n_image - cnt]
                image_modalitys = image_modalitys[:self.n_image - cnt]

            image_tensor = torch.tensor(image_datas, dtype=torch.float32)
            key_tensor = torch.tensor(image_keys, dtype=torch.long)
            mark_tensor = torch.tensor(image_marks, dtype=torch.long)
            modality_tensor = torch.tensor(image_modalitys, dtype=torch.long)

            image_shape = image_tensor.shape

            image_tensor = rearrange(image_tensor, "n h w d -> d n h w")
            image_tensor = resample_image(image_tensor, self.size, self.depth)
            image_tensor = rearrange(image_tensor, "d n h w -> n h w d")
            image_tensor = image_tensor.unsqueeze(1)

            if image_tensor.max()-image_tensor.min() == 0:
                print("warning: 0")
                print(link)
                image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
            image_fuse[cnt:cnt+image_shape[0]] = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
            key[cnt:cnt+image_shape[0]] = key_tensor
            mark[cnt:cnt+image_shape[0]] = mark_tensor
            modality[cnt:cnt+image_shape[0]] = modality_tensor

            cnt += image_shape[0]
        return {
            "images": image_fuse, 
            "mods": modality,
            "marks": mark, 
            "keys": key,
            "labels": labels,
            }
    
    def __len__(self):
        return len(self.data_dict)




if __name__ == "__main__":
   
    datasets = RadNet_Dataset_early("/home/qiaoyuzheng/MedVisionMulKey/DataPath/Train_anatomyWith3D.json",
                               "/home/qiaoyuzheng/MedVisionMulKey/DataPath/sorted_disease_label_dict.json",                               
                               level='articles',
                               num_classes=5569,
                               size=256,
                               depth=32,
                            )
    dataloader = DataLoader(
            datasets,
            batch_size=16,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )  
    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
        """check the shape"""