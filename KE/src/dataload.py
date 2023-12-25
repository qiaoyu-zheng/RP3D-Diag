import os
import cv2
import logging
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from itertools import cycle


from dataclasses import dataclass
from multiprocessing import Value

import braceexpand

import torch
import torchvision.datasets as datasets
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data import Dataset, DataLoader, BatchSampler,ConcatDataset,RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class ICD10_Dataset_test(Dataset):
    def __init__(self,icd_json_file,icd_level_json_file):
        with open(icd_json_file, 'r') as file:
            self.id_json_data = json.load(file)
        with open('/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/ICD_dict_1218.json','r') as file:
            ours_icd_data = json.load(file)
        self.ours_icd_list = list(set(list(ours_icd_data.values())))
        
        with open(icd_level_json_file, 'r') as file:
            self.id_level_data = json.load(file)
        self.icd_data_len = len(self.id_json_data)

    def __len__(self):
        return len(self.ours_icd_list)

    def get_text(self,input_dict):
        try:
            input_text = input_dict['description']
        except:
            print(input_dict)
        try:
            pos_text = ''.join(input_dict['Clinical_Information'])
        except:
            try:
                pos_text = random.choice(input_dict['Approximate_Synonyms'])
            except:
                try:
                    pos_text = random.choice(input_dict['Applicable_To'])
                except:
                    pos_text = input_text
                    # try:
                    #     pos_text = random.choice(input_dict['Includes'])
                    # except:
                    #     pos_text = input_text
        return input_text,pos_text
    
    def __getitem__(self, idx):
        select_icd = self.id_level_data['level3'][idx]
        input_text,pos_text = self.get_text(self.id_json_data[select_icd])
        cui = str(select_icd)
        
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        sample['cui'] = cui
        return sample

class ICD10_Dataset(Dataset):
    def __init__(self,icd_json_file,icd_level_json_file,train_level=3):
        with open(icd_json_file, 'r') as file:
            self.id_json_data = json.load(file)
        with open(icd_level_json_file, 'r') as file:
            self.id_level_data = json.load(file)
        with open('/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/train_icd_level.json') as file:
            self.id_level_hierachy_data = json.load(file)
        self.id_level_hierachy_dict={}
        for level1_key in self.id_level_hierachy_data.keys():
            for level2_key in self.id_level_hierachy_data[level1_key].keys():
                for level3_key in self.id_level_hierachy_data[level1_key][level2_key].keys():
                    self.id_level_hierachy_dict[level3_key] = list(self.id_level_hierachy_data[level1_key][level2_key][level3_key].keys())
        self.icd_data_len = len(self.id_json_data)
        self.train_level = train_level 

    def __len__(self):
        if self.train_level == 1:
            return int(len(self.id_level_data['level1'])) 
        elif self.train_level == 2:
            return int(len(self.id_level_data['level2'])) 
        elif self.train_level == 3:
            return int(len(self.id_level_data['level3'])) 
        elif self.train_level == 4:
            return int(len(self.id_level_data['level4'])) 
        elif self.train_level == 5:
            return int(len(self.id_level_data['level3']))
    
    def is_trainable(self,input_dict):
        if len(input_dict.keys()) <=2:
            return False
        else:
            return True
    
    def get_text(self,input_dict):
        try:
            input_text = input_dict['description']
        except:
            print(input_dict)
        try:
            pos_text = ''.join(input_dict['Clinical_Information'])
        except:
            try:
                pos_text = random.choice(input_dict['Approximate_Synonyms'])
            except:
                try:
                    pos_text = random.choice(input_dict['Applicable_To'])
                except:
                    pos_text = input_text
                    # try:
                    #     pos_text = random.choice(input_dict['Includes'])
                    # except:
                    #     pos_text = input_text
        return input_text,pos_text

    def __getitem__(self, idx):
        #first level "A00-B99"
        if self.train_level == 1:
            level1_ids = self.id_level_data['level1']
            select_icd = random.choice(level1_ids)
            while 1:
                if self.is_trainable(self.id_json_data[select_icd]):
                    break
                else:
                    select_icd = random.choice(level1_ids)
            input_text,pos_text = self.get_text(self.id_json_data[select_icd])
            cui = str(select_icd)
        elif self.train_level == 2:
            level2_ids = self.id_level_data['level2']
            select_icd = random.choice(level2_ids)
            while 1:
                if self.is_trainable(self.id_json_data[select_icd]):
                    break
                else:
                    select_icd = random.choice(level2_ids)
            input_text,pos_text = self.get_text(self.id_json_data[select_icd])
            cui = str(select_icd)
        elif self.train_level == 3:
            level3_ids = self.id_level_data['level3']
            select_icd = random.choice(level3_ids)
            # select_icd = self.id_level_data['level3'][idx]
            while 1:
                if self.is_trainable(self.id_json_data[select_icd]):
                    break
                else:
                    select_icd = random.choice(level3_ids)
            input_text,pos_text = self.get_text(self.id_json_data[select_icd])
            cui = str(select_icd)
        elif self.train_level == 4:
            level4_ids = self.id_level_data['level4']
            select_icd = random.choice(level4_ids)
            while 1:
                if self.is_trainable(self.id_json_data[select_icd]):
                    break
                else:
                    select_icd = random.choice(level4_ids)
            input_text,pos_text = self.get_text(self.id_json_data[select_icd])
            cui = str(select_icd)
        elif self.train_level == 5:
            level3_ids = self.id_level_data['level3']
            select_icd = random.choice(level3_ids)
            cui = str(select_icd)
            
            select_icd_list = self.id_level_hierachy_dict[select_icd]
            while 1:
                if len(select_icd_list) > 2:
                    break 
                else:
                    select_icd = random.choice(level3_ids)
                    select_icd_list = self.id_level_hierachy_dict[select_icd]
            input_icd = random.choice(select_icd_list)
            pos_icd = random.choice(select_icd_list)
            input_text = self.id_json_data[input_icd]['description']
            pos_text = self.id_json_data[pos_icd]['description']
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        sample['cui'] = cui
        return sample
        
class UMLS_ICD_Dataset(Dataset):
    def __init__(self,mrdef_csv_file, umls_kg_file, umls_cui_file,rp_json_file,ics_json_file):
        with open(ics_json_file, 'r') as file:
            self.ics_json_data = json.load(file)
        self.icd_list = list(self.ics_json_data.keys())
        
        self.mrdef_info = pd.read_csv(mrdef_csv_file)
        self.mrdef_cui_list = self.mrdef_info.iloc[:,0]
        self.mrdef_name_list = self.mrdef_info.iloc[:,1]
        self.mrdef_def_list = self.mrdef_info.iloc[:,2]

        self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_kg_source_list = self.umls_kg_info.iloc[:,0]
        self.umls_kg_target_list = self.umls_kg_info.iloc[:,1]
        self.umls_kg_edge_list = self.umls_kg_info.iloc[:,2]

        self.umls_cui_info = pd.read_csv(umls_cui_file)
        self.umls_cui_source_list = self.umls_cui_info.iloc[:,0]
        self.umls_cui_target_list = self.umls_cui_info.iloc[:,1]
        

        self.umls_data_len = len(self.umls_kg_info)
        self.mrdef_data_len = len(self.mrdef_info)
        print('UMLS data length: ',self.umls_data_len)
        print('MRDEF data length: ',self.mrdef_data_len)
        self.select_umls_ratio = self.umls_data_len/(self.umls_data_len+self.mrdef_data_len)

        with open(rp_json_file, 'r') as file:
            self.rp_json_data = json.load(file)
        self.rp_disease_list = list(self.rp_json_data.keys())
        self.rp_disease_len = len(self.rp_disease_list)
        self.select_rp_ratio = 0.3
        self.select_icd_ratio = 0.1
    
    def __len__(self):
        return int(self.umls_data_len) 
    
    def __getitem__(self, idx):
        if random.random() < self.select_rp_ratio:
            # select definition
            input_text = random.choice(self.rp_disease_list)
            if self.rp_json_data[input_text]['radiographic_features'] == "":
                pos_text = self.rp_json_data[input_text]['definition']
            elif random.random() < 0.5:
                pos_text = self.rp_json_data[input_text]['definition']
            else:
                pos_text = self.rp_json_data[input_text]['radiographic_features']
            if len(self.rp_json_data[input_text]['umls_entities']) == 0:
                cui = str(0)
            else:
                try:
                    cui = self.rp_json_data[input_text]['umls_entities'][0]["CUI"]
                except:
                    cui = str(0)
        elif random.random() < self.select_icd_ratio:
            select_icd = random.choice(self.icd_list)
            cui = select_icd
            select_input_idx = random.randint(0,len(self.ics_json_data[select_icd])-1)
            select_pos_idx = random.randint(0,len(self.ics_json_data[select_icd])-1)
            input_text = self.ics_json_data[select_icd][select_input_idx]['long_title']
            pos_text = self.ics_json_data[select_icd][select_pos_idx]['long_title']
        elif random.random() < self.select_umls_ratio:
            select_idx = random.randint(0,self.umls_data_len-1)
            text_h = self.umls_kg_source_list[select_idx]
            cui_h = self.umls_cui_source_list[select_idx]
            text_t = self.umls_kg_target_list[select_idx]
            cui_t = self.umls_cui_target_list[select_idx]
            text_r = self.umls_kg_edge_list[select_idx]
            if random.random()<0.5:
                input_text = text_h + ' [SEP] ' + text_r
                pos_text =  text_t
                cui = cui_t
            else:
                input_text = text_r + ' [SEP] ' + text_t
                pos_text =  text_h
                cui = cui_h
        else:
            select_idx = random.randint(0,self.mrdef_data_len-1)
            input_text = self.mrdef_name_list[select_idx]
            pos_text = self.mrdef_def_list[select_idx]
            cui = self.mrdef_cui_list[select_idx]
            
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        try: 
            if cui[0] == 'C':
                sample['cui'] = cui
            else:
                sample['cui'] = str(0)
        except:
            sample['cui'] = str(0)
        return sample

class UMLS_Dataset(Dataset):
    def __init__(self,mrdef_csv_file, umls_kg_file, umls_cui_file,rp_json_file):
        self.mrdef_info = pd.read_csv(mrdef_csv_file)
        self.mrdef_cui_list = self.mrdef_info.iloc[:,0]
        self.mrdef_name_list = self.mrdef_info.iloc[:,1]
        self.mrdef_def_list = self.mrdef_info.iloc[:,2]

        self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_kg_source_list = self.umls_kg_info.iloc[:,0]
        self.umls_kg_target_list = self.umls_kg_info.iloc[:,1]
        self.umls_kg_edge_list = self.umls_kg_info.iloc[:,2]

        self.umls_cui_info = pd.read_csv(umls_cui_file)
        self.umls_cui_source_list = self.umls_cui_info.iloc[:,0]
        self.umls_cui_target_list = self.umls_cui_info.iloc[:,1]
        

        self.umls_data_len = len(self.umls_kg_info)
        self.mrdef_data_len = len(self.mrdef_info)
        print('UMLS data length: ',self.umls_data_len)
        print('MRDEF data length: ',self.mrdef_data_len)
        self.select_umls_ratio = 0.5*self.umls_data_len/(self.umls_data_len+self.mrdef_data_len)

        with open(rp_json_file, 'r') as file:
            self.rp_json_data = json.load(file)
        self.rp_disease_list = list(self.rp_json_data.keys())
        self.rp_disease_len = len(self.rp_disease_list)
        self.select_rp_ratio = 0.3
        
    def __len__(self):
        return int(self.umls_data_len) 
    
    def __getitem__(self, idx):
        if random.random() < self.select_rp_ratio:
            # select definition
            input_text = random.choice(self.rp_disease_list)
            if self.rp_json_data[input_text]['radiographic_features'] == "":
                pos_text = self.rp_json_data[input_text]['definition']
            elif random.random() < 0.5:
                pos_text = self.rp_json_data[input_text]['definition']
            else:
                pos_text = self.rp_json_data[input_text]['radiographic_features']
            if len(self.rp_json_data[input_text]['umls_entities']) == 0:
                cui = str(0)
            else:
                try:
                    cui = self.rp_json_data[input_text]['umls_entities'][0]["CUI"]
                except:
                    cui = str(0)
        elif random.random() < self.select_umls_ratio:
            select_idx = random.randint(0,self.umls_data_len-1)
            text_h = self.umls_kg_source_list[select_idx]
            cui_h = self.umls_cui_source_list[select_idx]
            text_t = self.umls_kg_target_list[select_idx]
            cui_t = self.umls_cui_target_list[select_idx]
            text_r = self.umls_kg_edge_list[select_idx]
            if random.random()<0.5:
                input_text = text_h + ' [SEP] ' + text_r
                pos_text =  text_t
                cui = cui_t
            else:
                input_text = text_r + ' [SEP] ' + text_t
                pos_text =  text_h
                cui = cui_h
        else:
            select_idx = random.randint(0,self.mrdef_data_len-1)
            input_text = self.mrdef_name_list[select_idx]
            pos_text = self.mrdef_def_list[select_idx]
            cui = self.mrdef_cui_list[select_idx]
            
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        try: 
            if cui[0] == 'C':
                sample['cui'] = cui
            else:
                sample['cui'] = str(0)
        except:
            sample['cui'] = str(0)
        return sample
        
class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


# 自定义 BatchSampler
class CustomBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        dataset_probs = [0.3, 0.5, 0.1, 0.01]
        for _ in range(self.batch_size):
            selected_dataset = torch.multinomial(torch.tensor(dataset_probs), 1).item()
            indices = list(self.sampler)
            batch.extend(indices)
        yield batch


if __name__ == "__main__":
    icd_json_file = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/train_icd.json'
    icd_level_json_file = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/train_icd_level_list.json'
    mrdef_csv_file = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/MRDEF_name.csv'
    umls_kg_file = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/umls_kg.csv'
    umls_cui_file = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/umls_cui.csv'
    rparticle_data = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadCLIP/src_batch/data/train_rparticles.json'
    # for train_level in range(1,4):
    #     dataset = ICD10_Dataset(icd_json_file,icd_level_json_file,train_level)

    #     # 迭代数据集中的前几个样本
    #     for i in range(5):  # 这里选择迭代前5个样本，根据需要调整
    #         sample = dataset[i]  # 获取数据集中的第i个样本
    #         print(f"Sample {i + 1}:\n{sample}")
    
    dataset1 = ICD10_Dataset(icd_json_file,icd_level_json_file,train_level=1)
    dataset2 = ICD10_Dataset(icd_json_file,icd_level_json_file,train_level=2)
    dataset3 = ICD10_Dataset(icd_json_file,icd_level_json_file,train_level=3)
    dataset4 = ICD10_Dataset(icd_json_file,icd_level_json_file,train_level=4)
    dataset5 = UMLS_Dataset(mrdef_csv_file, umls_kg_file, umls_cui_file,rparticle_data)
    print(len(dataset1),len(dataset2),len(dataset3),len(dataset4),len(dataset5))
    #  22, 202, 1361,20830,1023493
    
    # concat_dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4])
    # print(len(concat_dataset))
    # num_samples = len(concat_dataset)
    # sampler = RandomSampler(concat_dataset)
    # batch_sampler = CustomBatchSampler(sampler, batch_size=4, drop_last=True)
    # dataloader = DataLoader(concat_dataset, batch_sampler=batch_sampler)
    
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch {batch_idx + 1}:\n")
    #     print(len(batch['cui']))
    # dataloader1 = DataLoader(dataset1, batch_size=4, shuffle=True)
    # dataloader2 = DataLoader(dataset2, batch_size=4, shuffle=True)
    # dataloader3 = DataLoader(dataset3, batch_size=4, shuffle=True)
    # dataloader4 = DataLoader(dataset4, batch_size=4, shuffle=True)
    # dataloader_list = [dataloader1,dataloader2,dataloader3,dataloader4]
    
    # iter_dataloader1 = cycle(iter(dataloader_list[0]))
    # iter_dataloader2 = cycle(iter(dataloader_list[1]))
    # iter_dataloader3 = cycle(iter(dataloader_list[2]))
    # iter_dataloader4 = cycle(iter(dataloader_list[3]))
    # # iter_dataloader5 = cycle(iter(dataloader_list[4]))

    # for _,i in enumerate(range(10)):
    #     if i % 5 == 0:
    #         #1/10000
    #         batch = next(iter_dataloader1)
    #     elif i % 4 == 0:
    #         #1/1000
    #         batch = next(iter_dataloader2)
    #     elif i % 3 == 0:
    #         #1/1000
    #         batch = next(iter_dataloader4)
    #     # elif i % 10 == 0:
    #     #     #1/1000
    #     #     batch = next(iter_dataloader3)
    #     # else:
    #     #     batch = next(iter_dataloader5)
    #     print(batch)