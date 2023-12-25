from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig,AutoTokenizer



class MedCPT_clinical(nn.Module):
    def __init__(self,bert_model_name: str):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def _get_bert_basemodel(self, bert_model_name):#12
        try:
            model = AutoModel.from_pretrained(bert_model_name)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")
        return model
    
    def encode_text(self, text):
        output = self.bert_model(**text)
        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = output.last_hidden_state[:, 0, :]
        embeds = F.normalize(embeds, dim=-1)
        return embeds
    
    def forward(self,text1,text2):      
        text1_features = self.encode_text(text1)
        text2_features = self.encode_text(text2)
        text1_features = F.normalize(text1_features, dim=-1)
        text2_features = F.normalize(text2_features, dim=-1)
        # print(text1_features.shape,text2_features.shape)  
        return text1_features, text2_features, self.logit_scale.exp()


if __name__ == "__main__":
    model = MedCPT_clinical(bert_model_name = 'ncbi/MedCPT-Query-Encoder')
    checkpoint = torch.load('/mnt/petrelfs/zhengqiaoyu.p/RadNet_KE/Test/epoch_state.pt',map_location='cpu')['state_dict']
    load_checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    missing, unexpect = model.load_state_dict(load_checkpoint, strict=False)
    print(missing)
    print(unexpect)
    
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    queries = ['Hypothalamic dysfunction']
    
    with torch.no_grad():
        # tokenize the queries
        encoded = tokenizer(
            queries, 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=256,
        )
        
        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model.encode_text(encoded)

        # print(embeds)
        print(embeds.size())