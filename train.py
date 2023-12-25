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
    """找到数组中最接近给定值的元素的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculate_multilabel_AUC(labels, logits, epoch, name, split):
    # """计算多标签分类的平均AUC"""
    # 初始化AUC列表
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
    os.makedirs(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}", exist_ok=True)
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/AUC_TPR_max", np.array(tpr_list[max_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/AUC_TPR_min", np.array(tpr_list[min_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/AUC_FPR_max", np.array(fpr_list[max_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/AUC_FPR_min", np.array(fpr_list[min_index]))
    # 画图
    plt.figure()

    # 假设 fpr_list 和 tpr_list 是从之前的计算中得到的
    fpr_max = fpr_list[max_index]
    tpr_max = tpr_list[max_index]
    fpr_min = fpr_list[min_index]
    tpr_min = tpr_list[min_index]
    # 创建一个共同的 FPR 域
    common_fpr = np.linspace(0, 1, 100)  # 例如，创建100个点

    # 插值得到 TPR 值
    interp_tpr_max = interp1d(fpr_max, tpr_max, kind='linear')(common_fpr)
    interp_tpr_min = interp1d(fpr_min, tpr_min, kind='linear')(common_fpr)

    # 绘制曲线
    plt.plot(common_fpr, interp_tpr_max, color='blue', lw=2, label='Max AUC (area = %0.2f)' % max_auc)
    plt.plot(common_fpr, interp_tpr_min, color='red', lw=2, label='Min AUC (area = %0.2f)' % min_auc)

    # 填充两条曲线之间的区域
    plt.fill_between(common_fpr, interp_tpr_min, interp_tpr_max, color='grey', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    os.makedirs(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/Figure/{name}/{epoch}/{split}", exist_ok=True)
    plt.savefig(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/Figure/{name}/{epoch}/{split}/ROC.png")

    #每个sample都可能会有多个标签，寻找每个sample里面的prob最大的10，20，30个index，看看这些index和标签的交集所占标签总数的比例

    return mean_auc, max_auc, max_index, min_auc, min_index



def calculate_mAP(labels, logits, epoch, name, split):
    """
    计算多标签分类的平均精度(mAP)。
    param labels:基于真值标签的二元矩阵(形状[B, cls])
    param logits:预测分数或logits的矩阵(shape [B, cls])
    return: mAP分数
    """
    # 类别数量
    # scores = np.exp(logits)
    scores = sigmoid(logits)
    n_classes = labels.shape[1]
    
    # 初始化AP_list
    AP_list = []
    precision_list = []
    recall_list = []
    
    # 对每个类别计算一个AP
    for class_i in range(n_classes):
        if len(np.unique(labels[:, class_i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels[:, class_i], scores[:, class_i])
        # Calculate average precision for this class
        AP = average_precision_score(labels[:, class_i], scores[:, class_i])
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # 计算所有类别的AP的平均
    mAP = np.mean(AP_list)
    max_ap = np.max(AP_list)
    min_ap = np.min(AP_list)
    max_index = AP_list.index(max_ap)
    min_index = AP_list.index(min_ap)
    os.makedirs(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}", exist_ok=True)
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/PR_Precision_max", np.array(precision_list[max_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/PR_Precision_min", np.array(precision_list[min_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/PR_Recall_max", np.array(recall_list[max_index]))
    np.save(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/NPY/{name}/{epoch}/{split}/PR_Recall_min", np.array(recall_list[min_index]))
    # 绘制精确率-召回率曲线
    plt.figure()
    plt.plot(recall_list[max_index], precision_list[max_index], color='blue', lw=2, label='Max AP (area = %0.2f)' % max_ap)
    plt.plot(recall_list[min_index], precision_list[min_index], color='red', lw=2, label='Min AP (area = %0.2f)' % min_ap)

    # 填充两条曲线之间的区域
    common_recall = np.linspace(0, 1, 100)  # 创建100个点
    interp_precision_max = interp1d(recall_list[max_index], precision_list[max_index], kind='linear', bounds_error=False, fill_value='extrapolate')(common_recall)
    interp_precision_min = interp1d(recall_list[min_index], precision_list[min_index], kind='linear', bounds_error=False, fill_value='extrapolate')(common_recall)
    plt.fill_between(common_recall, interp_precision_min, interp_precision_max, color='grey', alpha=0.3)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    os.makedirs(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/Figure/{name}/{epoch}/{split}", exist_ok=True)
    plt.savefig(f"/remote-home/share/data200/172.16.11.200/zhengqiaoyu/RadNet_KE/DataOutput/Figure/{name}/{epoch}/{split}/PR.png")
    
    return mAP

def calculate_mF1max_MCC(labels, logits):
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
    """
    对于每个样本，找到概率最高的k个索引，并计算它们与真实标签的交集所占的比例。
    :param predictions: 预测的logits，形状为[samples, classes]。
    :param labels: 真实的标签，形状也为[samples, classes]，元素为0或1。
    :param ks: 包含不同k值的列表，例如[10, 20, 30]。
    :return: 包含不同k值对应交集占比列表的字典。
    """
    # 使用sigmoid函数将logits转换为概率
    probabilities = 1 / (1 + np.exp(-predictions))
    # 初始化每个k值对应的交集占比列表的字典
    iou_scores = {k: [] for k in ks}
    # 对每个样本计算交集占比
    for sample_prob, sample_label in zip(probabilities, labels):
        # 真实标签为1的索引
        label_indices = np.where(sample_label == 1)[0]
        # 获取概率排序后的索引
        sorted_indices = np.argsort(sample_prob)[::-1]
        for k in ks:
            # 获取当前k值下的概率最高的k个索引
            topk_indices = sorted_indices[:k]
            # 计算交集
            intersection = np.intersect1d(topk_indices, label_indices).shape[0]
            # 计算并存储交集和标签总数的比例
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
    set_seed()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    start_class = training_args.start_class   #0
    end_class = training_args.end_class  #917
    num_classes = end_class - start_class
    backbone = training_args.backbone  #'resnet'
    level = training_args.level    # 'icd10s'
    depth = training_args.depth   # 16
    ltype = training_args.ltype  #'MultiLabel'
    augment = training_args.augment  # True
    fuse = training_args.fuse  # 'comb'
    ke = training_args.ke
    adapter = training_args.adapter
    n_image = training_args.n_image
    name_str = f"{backbone}_{level}_{depth}_{ltype}_{augment}_{fuse}_{n_image}"
    checkpoint = None if training_args.checkpoint == "None" else training_args.checkpoint  #  None
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
    print("Texts: ", texts)
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
            missing, unexpect = load_model(model, safetensor, strict=False)
        else:
            raise ValueError("Invalid safetensors!")
        print(f"Missing: {missing}")
        print(f"Unexpect: {unexpect}")
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