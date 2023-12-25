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
random.seed(42)

Modality_dict = {"CT":[1,2,3], "MRI":[4,5,6], "Ultrasound": [7], "X-ray": [8], "Annotated image": [9], "Fluoroscopy": [10], 
                 "Mammography": [11], "DSA (angiography)": [12], "Nuclear medicine": [13], "Barium": [14]}


def resample_image(image, target):
    depth = image.shape[-1]
    if depth == target:
        output_tensor = image
    else:
        # Upsample or Downsample
        output_tensor = torch.nn.functional.interpolate(image, size=(512, 512, target), mode='trilinear', align_corners=False)
    return output_tensor


class RadNet_Dataset_late(Dataset):
    def __init__(self, data_path, label_path, num_classes=5569, level='articles', depth=16):
        with open(data_path, 'r') as f:
            self.data_dict = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.level = level
        self.depth = depth
        self.link_list = list(self.data_dict.keys())

    
    def __getitem__(self, index):
        link = self.link_list[index]
        Modalities = self.data_dict[link]["Modalities"]
        Samples = self.data_dict[link]["Samples"]
        length = len(Modalities)
        randnum = random.randrange(0,length)

        #process label
        label_key_list = Samples[randnum][self.level]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)

        #process image
        npy_path = Samples[randnum]["npy_path"]
        image_datas = np.load(npy_path)
        image_datas = image_datas[:,0:1,:,:,:]
        image_shape = image_datas.shape
        assert image_shape[1] == 1
        if image_shape[-1] < 3 or (image_shape[-1] == 3 and (image_datas[0][:,:,:,0] == image_datas[0][:,:,:,1]).all()):
            self.dim = 1
        else:
            self.dim = 2
        dimension = torch.tensor([self.dim])
        modality = torch.tensor([0])
        randpick = random.randrange(0,image_shape[0])
        image_data = image_datas[randpick]
        image_tensor = torch.tensor(image_data, dtype=torch.float32)
        if self.dim == 1:
            image_tensor = image_tensor[:,:,:,0].unsqueeze(-1).repeat(1,1,1,self.depth)
        elif self.dim == 2:
            # image_data = resample_image(image_data,64)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = resample_image(image_tensor, self.depth)
            image_tensor = image_tensor[0,:,:,:,:]
        else:
            assert self.dim != 0
        if image_tensor.max()-image_tensor.min() == 0:
            print("warinig: 0")
            print(link)
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        return {
            "images": image, # (1, 512, 512, depth)
            "mods": modality,
            "dims": dimension, # (1)
            "labels": labels,
            }
    
    def __len__(self):
        return len(self.data_dict)
    
class RadNet_Dataset_early(Dataset):
    def __init__(self, data_path, label_path, num_classes=5569, level='articles', depth=16, n_image=14):
        with open(data_path, 'r') as f:
            self.data_dict = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.level = level
        self.depth = depth
        self.n_image = n_image
        self.link_list = list(self.data_dict.keys())

    
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
        image_fuse = torch.zeros((self.n_image,1,512,512,self.depth), dtype=torch.float32)
        modality = torch.zeros((self.n_image,1), dtype=torch.long)
        dimension = torch.zeros((self.n_image,1), dtype=torch.long)
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

            # process image
            # if Modalities[i] in repeat_cnt:
            #     print("Repeat", Modalities[i])
            # else:
            #     repeat_cnt[Modalities[i]] = 1
            if Modalities[i] == 'CT' or Modalities[i] == 'MRI':
                npy_path = Samples[i]["npy_path"]
                image_datas = np.load(npy_path)
                if image_datas.shape[0] >= 3:
                    image_datas = image_datas[:3]               
            else:
                npy_path = Samples[i]["npy_path"]
                image_datas = np.load(npy_path)
                randpick = random.randrange(0,image_datas.shape[0])
                image_datas = image_datas[randpick:randpick+1]
                # image_datas = np.expand_dims(image_datas[randpick], axis=0)
            # assert (image_datas[:,0] == image_datas[:,1]).all()
            image_datas = image_datas[:,0:1,:,:,:]

            if cnt + image_datas.shape[0] > self.n_image:
                image_datas = image_datas[:self.n_image - cnt]

            image_shape = image_datas.shape

            assert image_shape[1] == 1
            mod = 0
            if image_shape[-1] < 3 or (image_shape[-1] == 3 and (image_datas[0][:,:,:,0] == image_datas[0][:,:,:,1]).all()):
                mod = 1
            else:
                mod = 2

            for j in range(image_shape[0]):
                dimension[cnt+j] = torch.tensor([mod])
                modality[cnt+j] = torch.tensor(Modality_dict[Modalities[i]][j])

            image_tensor = torch.tensor(image_datas, dtype=torch.float32)
            if mod == 1: 
                image_tensor = image_tensor[:,:,:,:,0].unsqueeze(-1).repeat(1,1,1,1,self.depth)
                # image_tensor = image_tensor[:,:,:,:,0:1].expand(-1,-1,-1,-1,self.depth)
            elif mod == 2:
                image_tensor = resample_image(image_tensor, self.depth)
                # image_tensor = image_tensor[:,:,:,:,0].unsqueeze(-1).repeat(1,1,1,1,self.depth)
            else:
                assert mod != 0
            if image_tensor.max()-image_tensor.min() == 0:
                print("warning: 0")
                image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
            image_fuse[cnt:cnt+image_shape[0]] = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
            cnt += image_shape[0]
        # return torch.randn((1,2))
        return {
            "images": image_fuse,
            "mods": modality,
            'dims': dimension,
            "labels": labels,
            }   
    def __len__(self):
        return len(self.data_dict)


if __name__ == "__main__":
    # with open("/mnt/petrelfs/zhengqiaoyu.p/MedVisionCLF/Datapath/radio3d_test_v1.json", 'r') as f:
    #     list_data = json.load(f)
    # print("shit")
    datasets = RadNet_Dataset_early("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision/DataPath/radio_3d_case_level_link_dict_final_all_train.json",
                               "/remote-home/share/data200/172.16.11.200/zhengqiaoyu/MedVision/DataPath/sorted_icd10_label_dict.json",
                               level='articles',
                               num_classes=5569,
                               depth=32,
                               n_image=6
                            )
    dataloader = DataLoader(
            datasets,
            batch_size=4,
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
    

