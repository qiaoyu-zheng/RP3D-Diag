from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import evaluate
from dataclasses import dataclass, field
from Dataset.dataset import RadNet_Dataset_early, RadNet_Dataset_late
from Model.modelRadNet import RadNet
import numpy as np
import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
from functools import partial
import wandb
import sys
import random
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from transformers import AutoModel,BertConfig,AutoTokenizer
from safetensors.torch import load_model
import json
sys.path.append("Path/to/RP3D-Diag")
sys.path.append("Path/to/RP3D-Diag/src")

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
    output_dir: Optional[str] = field(default="/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/results/output1")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    # My new parameters
    start_class: int = field(default=0)
    end_class: int = field(default=917)
    backbone: str = field(default='resnet')
    level: str = field(default='icd10s')
    depth: int = field(default=16)
    ltype: str = field(default='BCEFocal')
    augment: bool = field(default=True)
    n_image: int = field(default=14)
    fuse: str = field(default='late')
    ke: str = field(default=False)
    adapter: str = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    safetensor: Optional[str] = field(default=None)

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images, mods, dims, labels = tuple([instance[key] for instance in instances] for key in ('images', 'mods', 'dims', 'labels'))
        images = torch.cat([_.unsqueeze(0) for _ in images],dim  = 0)
        mods = torch.cat([_.unsqueeze(0) for _ in mods],dim  = 0)
        dims = torch.cat([_.unsqueeze(0) for _ in dims],dim  = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels],dim  = 0)
        
        return_dic = dict(
            image_x=images,
            mods=mods,
            dims=dims,
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
    """
    Calculate AUC for each class and return the average AUC.
    """
    # Initialize AUC list
    auc_list = []
    fpr_list = []
    tpr_list = []
    scores = sigmoid(logits)
    # logits = np.exp(logits)
    # 遍历每一列（即每个类别）
    for i in range(labels.shape[1]):
        # 计算当前类别的AUC
        # 注意：roc_auc_score 预期的输入是二进制标签和概率或者分数
        # logits 需要通过sigmoid函数转换成概率
        if len(np.unique(labels[:, i])) < 2:
            continue
        auc = roc_auc_score(labels[:, i], scores[:, i])
        auc_list.append(auc)

        fpr, tpr, _ = roc_curve(labels[:, i], scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    # 计算所有类别AUC的平均值
    max_auc = np.max(auc_list)
    mean_auc = np.mean(auc_list)
    min_auc = np.min(auc_list)
    max_index = auc_list.index(max_auc)
    min_index = auc_list.index(min_auc)

    return mean_auc, max_auc, max_index, min_auc, min_index



def calculate_mAP(labels, logits, epoch, name, split):
    """
    Calculate mAP for each class and return the average mAP.
    """
    scores = sigmoid(logits)
    n_classes = labels.shape[1]
    
    # initialize average precision list
    AP_list = []
    precision_list = []
    recall_list = []
    
    # calculate average precision for each class
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        # Calculate average precision for this class
        AP = average_precision_score(labels[:, class_i], scores[:, class_i])
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # calculate average precision for all classes
    mAP = np.mean(AP_list)
    max_ap = np.max(AP_list)
    min_ap = np.min(AP_list)
    max_index = AP_list.index(max_ap)
    min_index = AP_list.index(min_ap)
    
    return mAP

def calculate_mF1max_MCC(labels, logits):
    """
    Calculate mF1max and MCC for each class and return the average mF1max and MCC.
    """
    scores = sigmoid(logits)
    n_classes = labels.shape[1]

    F1max_list = []
    MCC_list = []

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

    return np.mean(F1max_list), np.mean(MCC_list)

def calculate_mRecall_FPR(labels, logits, fpr_points):
    """
    Calculate mRecall@FPR for each class and return the average mRecall@FPR.
    """
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
        # 在这里保存评估信息
        self.eval_epoch = state.epoch

def eval_strategy(labels, logits, epoch, name, split):
    mAUC, max_auc, max_index, min_auc, min_index = calculate_multilabel_AUC(labels, logits, epoch, name, split)
    mAP_CW = calculate_mAP(labels, logits, epoch, name, split)
    mF1max, mMCC = calculate_mF1max_MCC(labels, logits)
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
    loss,logits,labels = predictions
    logits = np.clip(logits,-60,60)
    # head icd: [:165]; disease: [:85]
    # body icd: [165:395]; disease: [85:555]
    # tail: [395:]; disease: [555:]
    
    if level == 'icd10s':
        labels_head, logits_head = labels[:,:165], logits[:,:165]
        labels_body, logits_body = labels[:,165:395], logits[:,165:395]
        labels_tail, logits_tail = labels[:,395:], logits[:,395:]
    elif level == 'articles':
        labels_head, logits_head = labels[:,:85], logits[:,:85]
        labels_body, logits_body = labels[:,85:555], logits[:,85:555]
        labels_tail, logits_tail = labels[:,555:], logits[:,555:]
    else:
        raise ValueError("Invalid level in compute metrics!")

    evalRes_head = eval_strategy(labels_head, logits_head, epoch, name, 'head')
    evalRes_body = eval_strategy(labels_body, logits_body, epoch, name, 'body')
    evalRes_tail = eval_strategy(labels_tail, logits_tail, epoch, name, 'tail')
    
    metrics = {
        "loss": np.mean(loss),
        'mAUC_head': evalRes_head['mAUC'],
        'mAP_head': evalRes_head['mAP_CW'],
        'mF1max_head': evalRes_head['mF1max'],
        'mMCC_head': evalRes_head['mMCC'],
        'mR@F0.01_head': evalRes_head['mR@F0.01'],
        'mR@F0.05_head': evalRes_head['mR@F0.05'], 
        'mR@F0.1_head': evalRes_head['mR@F0.1'],
        'mR@F0.2_head': evalRes_head['mR@F0.2'],
        'mR@F0.5_head': evalRes_head['mR@F0.5'], 
        'mAUC_body': evalRes_body['mAUC'],
        'mAP_body': evalRes_body['mAP_CW'],
        'mF1max_body': evalRes_body['mF1max'],
        'mMCC_body': evalRes_body['mMCC'],
        'mR@F0.01_body': evalRes_body['mR@F0.01'],
        'mR@F0.05_body': evalRes_body['mR@F0.05'], 
        'mR@F0.1_body': evalRes_body['mR@F0.1'],
        'mR@F0.2_body': evalRes_body['mR@F0.2'],
        'mR@F0.5_body': evalRes_body['mR@F0.5'], 
        'mAUC_tail': evalRes_tail['mAUC'],
        'mAP_tail': evalRes_tail['mAP_CW'],
        'mF1max_tail': evalRes_tail['mF1max'],
        'mMCC_tail': evalRes_tail['mMCC'],
        'mR@F0.01_tail': evalRes_tail['mR@F0.01'],
        'mR@F0.05_tail': evalRes_tail['mR@F0.05'], 
        'mR@F0.1_tail': evalRes_tail['mR@F0.1'],
        'mR@F0.2_tail': evalRes_tail['mR@F0.2'],
        'mR@F0.5_tail': evalRes_tail['mR@F0.5'],   
    }
    return metrics
    
def main():
    """
    Parameter Settings:
    start_class, end_class for ICD10s and Disorders are relatively 0, 917 and 0,5569
    num_classes = end_class - start_class
    bakbone can be choosed from ['resnet', 'vit']
    level can be choosed from ['icd10s', 'articles'] where icd10s denotes ICD-10-CM and articles denotes Disorders
    depth can be choosed from [16, 24, 32]
    ltype can be choosed from ['MultiLabel', 'BCEFocal', 'MultiSoft', 'Asymmetric']
    augment can be choosed from [True, False]
    fuse can be choosed from ['early', 'late']
    ke can be choosed from [True, False]
    adapter can be choosed from [True, False]
    n_image can be choosed from [6,14]
    checkpoint can be choosed from [None, path]
    safetensor can be choosed from [None, path]
    """
    set_seed()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    start_class = training_args.start_class  
    end_class = training_args.end_class  
    num_classes = end_class - start_class
    backbone = training_args.backbone  
    level = training_args.level    
    depth = training_args.depth 
    ltype = training_args.ltype 
    augment = training_args.augment  
    fuse = training_args.fuse 
    ke = training_args.ke
    adapter = training_args.adapter
    n_image = training_args.n_image
    name_str = f"{backbone}_{level}_{depth}_{ltype}_{augment}_{fuse}_{n_image}"
    checkpoint = None if training_args.checkpoint == "None" else training_args.checkpoint  
    safetensor = None if training_args.safetensor == "None" else training_args.safetensor
    print(name_str)
    print("Setup Data")
    train_path = '/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/radio_3d_case_level_link_dict_final_all_new_train.json'
    eval_path = '/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/radio_3d_case_level_link_dict_final_all_new_eval.json'
    label_path = "/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/sorted_icd10_label_dict_new.json" if level == 'icd10s' else "/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/sorted_disease_label_dict.json"
    if fuse == 'late':
        train_datasets = RadNet_Dataset_late(train_path, label_path, num_classes, level, depth)
        eval_datasets = RadNet_Dataset_late(eval_path, label_path, num_classes, level, depth)
    elif fuse == 'early':
        train_datasets = RadNet_Dataset_early(train_path, label_path, num_classes, level, depth, n_image)
        eval_datasets = RadNet_Dataset_early(eval_path, label_path, num_classes, level, depth, n_image)
    else:
        raise ValueError("Invalid split value!")
    partial_compute_metrics = partial(compute_metrics, level=level, name=name_str)

    print("Tokenizing!")
    if level == 'icd10s':
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        labels = list(label_dict.keys())
        with open("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataPath/icd10_desription.json", 'r') as f:
            icd_des = json.load(f)
        texts = [icd_des[key] for key in labels]
    elif level == 'articles':
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        texts = list(label_dict.keys())
    else:
        raise ValueError("Invalid ltype value!")
    # print("Texts: ", texts)
    tokenizer = AutoTokenizer.from_pretrained("/remote-home/share/data200/172.16.11.200/zhengqiaoyu/pretrained")
    encoded = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        return_tensors='pt', 
        max_length=256,
    )
    print("Setup Model")
    model = RadNet(num_cls=num_classes, backbone=backbone, depth=depth, ltype=ltype, augment=augment, fuse=fuse, ke=ke, encoded=encoded, adapter=adapter)
    if safetensor is not None:
        if safetensor.endswith('.bin'):
            pretrained_weights = torch.load(safetensor)
            missing, unexpect = model.load_state_dict(pretrained_weights,strict=False)
        elif safetensor.endswith('.safetensors'):
            missing, unexpect = load_model(model, safetensor, strict=True)
        else:
            raise ValueError("Invalid safetensors!")
    print(missing)
    print(unexpect)
    trainer = Trainer(
        model=model,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=training_args,
        data_collator=DataCollator(),
        compute_metrics=partial_compute_metrics,
    )
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "200"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.add_callback(metrics_callback)
    
    print(trainer.evaluate())

if __name__ == "__main__":
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()