from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import evaluate
from dataclasses import dataclass, field
from Dataset.dataset import RadNet_Dataset_late, ChestXDet10, CheXpert, COVID19_Rad, IU_Xray, PadChest, BrainTumor, BrainTumor17, CT_Kidney, KneeMRI, MURA, POCUS, VinDr_Spine, CC_CCII
from Model.modelRadNet import RadNet
import numpy as np
import torch
# from My_Trainer.trainer import Trainer
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process
# import evaluate
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef, accuracy_score, confusion_matrix
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
    output_dir: Optional[str] = field(default=".../src/Finetune/Logout")
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
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculate_multilabel_AUC(labels, logits, epoch, name, split):
    auc_list = []
    fpr_list = []
    tpr_list = []
    scores = sigmoid(logits)
    # logits = np.exp(logits)
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            continue
        auc = roc_auc_score(labels[:, i], scores[:, i])
        auc_list.append(auc)

        fpr, tpr, _ = roc_curve(labels[:, i], scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    max_auc = np.max(auc_list)
    mean_auc = np.mean(auc_list)
    min_auc = np.min(auc_list)
    max_index = auc_list.index(max_auc)
    min_index = auc_list.index(min_auc)

    return mean_auc, max_auc, max_index, min_auc, min_index



def calculate_mAP(labels, logits, epoch, name, split):
    # scores = np.exp(logits)
    scores = sigmoid(logits)
    n_classes = labels.shape[1]
    
    AP_list = []
    precision_list = []
    recall_list = []
    
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        # Calculate average precision for this class
        AP = average_precision_score(labels[:, class_i], scores[:, class_i])
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
    
    mAP = np.mean(AP_list)
    max_ap = np.max(AP_list)
    min_ap = np.min(AP_list)
    max_index = AP_list.index(max_ap)
    min_index = AP_list.index(min_ap)
    
    return mAP

def calculate_mF1max_MCC(labels, logits):
    scores = sigmoid(logits)
    n_classes = labels.shape[1]

    F1max_list = []
    MCC_list = []
    ACC_list = []
    TPR_list = []
    FPR_list = []

    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, thresholds = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores[np.isnan(f1_scores)] = 0 
        max_f1_index = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_index]
        max_f1_threshold = thresholds[max_f1_index]
        F1max_list.append(max_f1)
        pred_class = (scores[:, class_i] >= max_f1_threshold).astype(int)
        mcc = matthews_corrcoef(labels[:, class_i], pred_class)
        MCC_list.append(mcc)
        acc = accuracy_score(labels[:, class_i], pred_class)
        ACC_list.append(acc)
        cm = confusion_matrix(labels[:, class_i], pred_class)
        TN, FP, FN, TP = cm.ravel()

        TPR = TP / (TP + FN)
        TPR_list.append(TPR)
        FPR = FP / (FP + TN)
        FPR_list.append(FPR)

    return np.mean(F1max_list), np.mean(MCC_list), np.mean(ACC_list), np.mean(TPR_list), np.mean(FPR_list)

def calculate_mRecall_FPR(labels, logits, fpr_points):
    scores = sigmoid(logits)
    n_classes = labels.shape[1]

    macro_recall_at_fpr = {fpr: [] for fpr in fpr_points}
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(labels[:, class_i], scores[:, class_i])

        for point in fpr_points:
            idx = find_nearest(fpr, point)
            macro_recall_at_fpr[point].append(tpr[idx])

    average_recall_at_fpr = {fpr: np.mean(recalls) for fpr, recalls in macro_recall_at_fpr.items()}
    return average_recall_at_fpr

def retrieval_mAP(labels, logits):
    AP_list = []
    for i in range(labels.shape[0]):
        if len(np.unique(labels[i,:])) < 2:
            continue
        logits_sample = logits[i]
        labels_sample = labels[i]
        logits_pos = logits_sample[labels_sample == 1]

        logits_sorted = np.sort(logits_sample)[::-1]
        logits_pos_sort = np.sort(logits_pos)[::-1]

        indices_pos = np.arange(1,logits_pos.shape[0]+1)
        indices_all = np.array([np.where(logits_sorted == element)[0][0] for element in logits_pos_sort]) + 1
        score_array = indices_pos / indices_all
        score = np.mean(score_array)
        AP_list.append(score)
    return np.mean(AP_list)


def top_k_intersection_over_union(predictions, labels, ks):
    probabilities = 1 / (1 + np.exp(-predictions))
    iou_scores = {k: [] for k in ks}
    for sample_prob, sample_label in zip(probabilities, labels):
        label_indices = np.where(sample_label == 1)[0]
        sorted_indices = np.argsort(sample_prob)[::-1]
        for k in ks:
            topk_indices = sorted_indices[:k]
            intersection = np.intersect1d(topk_indices, label_indices).shape[0]
            union = label_indices.shape[0]
            iou_score = intersection / union if union > 0 else 0
            iou_scores[k].append(iou_score)
    return iou_scores

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.eval_epoch = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_epoch = state.epoch

def eval_strategy(labels, logits, epoch, name, split):
    mAUC, max_auc, max_index, min_auc, min_index = calculate_multilabel_AUC(labels, logits, epoch, name, split)
    mAP_CW = calculate_mAP(labels, logits, epoch, name, split)
    mF1max, mMCC, mACC, mTPR, mFPR = calculate_mF1max_MCC(labels, logits)
    average_precision_score = calculate_mRecall_FPR(labels, logits, [0.01, 0.05, 0.1, 0.2, 0.5])
    mean_recall_at1 = average_precision_score[0.01]
    mean_recall_at5 = average_precision_score[0.05]
    mean_recall_at10 = average_precision_score[0.1]
    mean_recall_at20 = average_precision_score[0.2]
    mean_recall_at50 = average_precision_score[0.5]
    # return mAUC, mAP_CW, mF1max, mMCC, mean_recall_at1, mean_recall_at5, mean_recall_at10, mean_recall_at20, mean_recall_at50
    return {
        'mAUC': mAUC,
        'mAP_CW': mAP_CW,
        'mF1max': mF1max,
        'mMCC': mMCC,
        'mACC': mACC,
        'mTPR': mTPR,
        'mFPR': mFPR,
        'mR@F0.01': mean_recall_at1,
        'mR@F0.05': mean_recall_at5, 
        'mR@F0.1': mean_recall_at10,
        'mR@F0.2': mean_recall_at20,
        'mR@F0.5': mean_recall_at50, 
    }

metrics_callback = MetricsCallback()

def compute_metrics(eval_preds, level, name):
    epoch = metrics_callback.eval_epoch
    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids
    loss,loss_attn,loss_cls,logits,labels = predictions
    logits = np.clip(logits,-60,60)


    evalRes_head = eval_strategy(labels, logits, epoch, name, 'head')

    
    metrics = {
        "loss": np.mean(loss),
        "loss_attn": np.mean(loss_attn),
        "loss_cls": np.mean(loss_cls),
        'mAUC_head': evalRes_head['mAUC'],
        'mAP_head': evalRes_head['mAP_CW'],
        'mF1max_head': evalRes_head['mF1max'],
        'mMCC_head': evalRes_head['mMCC'],
        'mACC_head': evalRes_head['mACC'],
        'mTPR_head': evalRes_head['mTPR'],
        'mFPR_head': evalRes_head['mFPR'],
        'mR@F0.01_head': evalRes_head['mR@F0.01'],
        'mR@F0.05_head': evalRes_head['mR@F0.05'], 
        'mR@F0.1_head': evalRes_head['mR@F0.1'],
        'mR@F0.2_head': evalRes_head['mR@F0.2'],
        'mR@F0.5_head': evalRes_head['mR@F0.5'],
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
    root_path = ".../src/Finetune/DataPath/.../"
    train_path = f"{root_path}train.json"
    eval_path = f"{root_path}test.json"
    label_path = f"{root_path}label_dict.json"
    
    if fuse == 'late':
        print("Late!")
        train_datasets = BrainTumor(train_path, label_path, num_classes, size, depth, partial=1) #here you may need to change the dataset name
        eval_datasets = BrainTumor(eval_path, label_path, num_classes, size, depth)
    
    partial_compute_metrics = partial(compute_metrics, level=level, name=name_str)
    """
    print("Tokenizing!")
    if level == 'ICD':
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        labels = list(label_dict.keys())
        with open(".../src/Finetune/DataPath/icd10_desription.json", 'r') as f:
            icd_des = json.load(f)
        texts = [icd_des[key] for key in labels]
    elif level == 'articles':
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        texts = list(label_dict.keys())
    else:
        raise ValueError("Invalid ltype value!")
    print("Texts: ", texts)
    tokenizer = AutoTokenizer.from_pretrained("/remote-home/share/data-H800-200/zhengqiaoyu/pretrained")
    encoded = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        return_tensors='pt', 
        max_length=256,
    )
    print("Setup Model")
    """
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
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    print(trainer.evaluate())

if __name__ == "__main__":
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()