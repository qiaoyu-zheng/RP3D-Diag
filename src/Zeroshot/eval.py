from tqdm import tqdm
import numpy as np
import torch
# from My_Trainer.trainer import Trainer
# import evaluate
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef, accuracy_score, confusion_matrix
import os
import json

    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def find_nearest(array, value):
    """找到数组中最接近给定值的元素的索引"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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



logits = np.load(".../src/DataOutput/logits_eval.npy")
labels = np.load(".../src/DataOutput/labels_eval.npy")

selected_cls = [377, 1, 48, 0] #here please choose the classes, take brain-tumor dataset as an example

labels_slected, logits_selected = labels[:, selected_cls], logits[:, selected_cls]
mF1max, mMCC, mACC, mTPR, mFPR = calculate_mF1max_MCC(labels_slected, logits_selected)

print(mF1max, mMCC, mACC)

