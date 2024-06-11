from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import evaluate
from dataclasses import dataclass, field
from Dataset.dataset import RadNet_Dataset_late, ChestXDet10, CheXpert, COVID19_Rad, IU_Xray, PadChest, BrainTumor, BrainTumor17, CT_Kidney, KneeMRI, MURA, POCUS, VinDr_Spine
from Model.modelRadNet import RadNet
import numpy as np
import torch
# from My_Trainer.trainer import Trainer
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process
# import evaluate
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
from functools import partial
import wandb
import sys
import random
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from transformers import AutoModel,BertConfig,AutoTokenizer
from safetensors.torch import load_model
from Model.vitFuse import vitFuse
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class ModelArguments:
    tokenizer_name: Optional[str] = field(
        default='malteos/PubMedNCL', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default = False)
    per_device_train_batch_size: int = field(default = 32)
    per_device_eval_batch_size: int = field(default = 32)
    output_dir: Optional[str] = field(default=".../src/Zeroshot/logout")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_drop_last: bool = field(default=True)

    # My new parameters
    start_class: int = field(default=0)
    end_class: int = field(default=917)
    backbone: str = field(default='resnet')
    level: str = field(default='ICD')
    size: int = field(default=512)
    depth: int = field(default=16)
    ltype: str = field(default='BCEFocal')
    augment: bool = field(default=True)
    n_image: int = field(default=14)
    n_aug: int = field(default=3)
    prob: float = field(default=0.5)
    dim: int = field(default=2)
    hid_dim: int = field(default=2048)
    fuse: str = field(default='late')
    mix: str = field(default=False)
    ke: str = field(default=False)
    adapter: str = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    safetensor: Optional[str] = field(default=None)

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images, mods, marks, keys, labels = tuple([instance[key] for instance in instances] for key in ('images', 'mods', 'marks', 'keys', 'labels'))
        images = torch.cat([_.unsqueeze(0) for _ in images],dim  = 0)
        mods = torch.cat([_.unsqueeze(0) for _ in mods],dim  = 0)
        marks = torch.cat([_.unsqueeze(0) for _ in marks],dim  = 0)
        keys = torch.cat([_.unsqueeze(0) for _ in keys],dim  = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels],dim  = 0)
        
        return_dic = dict(
            image_x=images,
            mods=mods,
            marks=marks,
            keys=keys,
            labels=labels,
        )
        return return_dic
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def find_nearest(array, value):
    """找到数组中最接近给定值的元素的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.eval_epoch = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # 在这里保存评估信息
        self.eval_epoch = state.epoch


metrics_callback = MetricsCallback()

def compute_metrics(eval_preds, level, name):
    epoch = metrics_callback.eval_epoch
    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids
    loss,loss_attn,loss_cls,logits,labels = predictions
    logits = np.clip(logits,-60,60)
    np.save(".../src/DataOutput/logits_eval.npy", logits)
    np.save(".../src/DataOutput/labels_eval.npy", labels)
    
    metrics = {
        "loss": np.mean(loss),
        "loss_attn": np.mean(loss_attn),
        "loss_cls": np.mean(loss_cls),
    }
    return metrics
    
def main():
    set_seed()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    start_class = training_args.start_class   #0
    end_class = training_args.end_class  #85
    num_classes = end_class - start_class
    backbone = training_args.backbone  #'resnet'
    level = training_args.level    # 'ICD'
    size = training_args.size    # 512
    depth = training_args.depth   # 16
    ltype = training_args.ltype  #'MultiLabel'
    augment = training_args.augment  # True
    fuse = training_args.fuse  # 'comb'
    mix = training_args.mix #False
    ke = training_args.ke
    adapter = training_args.adapter
    n_image = training_args.n_image
    n_aug = training_args.n_aug
    prob = training_args.prob
    dim =training_args.dim
    hid_dim = training_args.hid_dim
    name_str = f"{backbone}_{level}_{depth}_{ltype}_{augment}_{fuse}_{n_image}"
    checkpoint = None if training_args.checkpoint == "None" else training_args.checkpoint  #  None
    safetensor = None if training_args.safetensor == "None" else training_args.safetensor
    print(name_str)
    print("Setup Data")
    root_path = ".../src/Finetune/DataPath/Brain-Tumor/"
    train_path = f"{root_path}train.json"
    eval_path = f"{root_path}test.json"
    label_path = f"{root_path}label_dict.json"
  

    if fuse == 'late':
        print("Late!")
        train_datasets = BrainTumor(train_path, label_path, num_classes, size, depth)
        eval_datasets = BrainTumor(eval_path, label_path, num_classes, size, depth)
        # train_datasets = RadNet_Dataset_late(train_path, label_path, num_classes, level, size, depth)
        # eval_datasets = RadNet_Dataset_late(eval_path, label_path, num_classes, level, size, depth)
    
    partial_compute_metrics = partial(compute_metrics, level=level, name=name_str)

    encoded = None
    model = RadNet(num_cls=num_classes, backbone=backbone, hid_dim=hid_dim, size=size, ltype=ltype, augment=augment, fuse=fuse)
    if safetensor is not None:
        if safetensor.endswith('.bin'):
            pretrained_weights = torch.load(safetensor)
            missing, unexpect = model.load_state_dict(pretrained_weights,strict=False)
        elif safetensor.endswith('.safetensors'):
            missing, unexpect = load_model(model, safetensor, strict=False)
        else:
            raise ValueError("Invalid safetensors!")
        print(f"Missing: {missing}")
        print(f"Unexpect: {unexpect}")
        
    # for param in model.resnet2D.parameters():
    #     param.requires_grad = False
    
    trainer = Trainer(
        model=model,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=training_args,
        data_collator=DataCollator(),
        compute_metrics=partial_compute_metrics,
    )
    # trainer.integrate_wandb()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "200"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.add_callback(metrics_callback)
    print(trainer.evaluate())

if __name__ == "__main__":
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()