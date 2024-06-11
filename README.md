# RP3D-Diag

The official codes for paper "Large-scale Long-tailed Disease Diagnosis on Radiology Images"

[ArXiv Version](https://arxiv.org/abs/2312.16151)

In this paper, we build up an academically accessible, large-scale diagnostic dataset, present a knowledge enhanced model architecture that enables to process arbitrary number of input scans from various imaging modalities, and initialize a new benchmark for multi-modal multi-anatomy long-tailed diagnosis. Our method shows superior results on it. Additionally, our final model serves as a pre-trained model, and can be finetuned to benefit diagnosis on various external datasets.

## Dataset

**Overview of RP3D-DiagDS.** There are **40,936 cases (195,010 scans)** across 7 human anatomy regions and 9 diverse modalities covering **5569 disorders** mapped into **931 ICD-10-CM codes**.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/RP3D-DiagDS.png"/>



The train/test split strategy and label csv files can be found in HuggingFace repository [RP3D-DiagDS](https://huggingface.co/datasets/QiaoyuZheng/RP3D-DiagDS). (Under updating).

## Model

The overview of our method. Three parts demonstrate our proposed visual encoders and fusion module, together
with the knowledge enhancement strategy respectively. a, The three types of vision encoder, i.e., ResNet-based, ViT-based , and
ResNet-ViT-mixing. b, The architecture of the fusion module. The figure shows the transformer-based fusion module, enabling
case-level information fusion. c, The knowledge enhancement strategy. We first pre-train a text encoder with extra medical
knowledge with contrastive learning, leveraging synonyms, descriptions, and hierarchy. Then we view the text embedding as a
natural classifier to guide the diagnosis classification.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/RP3D-DiagModel.png"/> 

The model checkpoint can also be found in HuggingFace repository [RadDiag](https://huggingface.co/QiaoyuZheng/RP3D-DiagModel). (Under updating).

## Setup

### Environment

To Install the python environments:
```
pip install -r requirements.txt
```

### Data Preparation

* See the readme.txt in src/Datapath and download files from HuggingFace.
* Modify all the paths (files in DataPath,all the python files and bash files) to your own.
* The DataPath folder shuold be like:

```
src
|   DataPath
|   |    train.json
|   |    test.json
|   |    aug.json
|   |    disorder_label_dict.json
|   |    icd10_label_dict.json
|   |    xxxCheckpoint
|   |    |    pytorch.model.bin 
|   |    |    ......
```

## Evaluation

1. Modify eval.sh (OUTDIR, CHECKPOINT)
2. Run in terminal:  `bash eval.sh`

## Training

### From Scratch

1. Modify train.sh (OUTDIR, LEVEL, DEPTH, FUSE, KE...)
2. Run in terminal: `bash train.sh`

### Load Checkpoint

1. Modify train.sh (SAFETENSOR), the path should be the path for pytorch.model.bin, Modify CHECKPOINT to "None".
2. Run in terminal: `bash train.sh`

## Benchmark

**Classification results on Disorders and ICD-10-CM levels.** In this table **Fusion Module** and **Knowledge Enhancement** strategy are both used. We report the results on Head/Medium/Tail class sets separately.

| Granularity | Class |  AUC  |  AP  |  F1  |  MCC  | R@0.01 | R@0.05 | R@0.1 |
| :---------: | :----: | :---: | :---: | :---: | :---: | :----: | :----: | :---: |
|  Disorders  |  Head  | 94.24 | 15.13 | 25.68 | 26.71 | 37.06 | 66.55 | 81.37 |
|  Disorders  | Medium | 94.69 | 12.38 | 20.64 | 24.07 | 31.52 | 65.34 | 78.73 |
|  Disorders  |  Tail  | 90.64 | 9.25 | 9.89 | 14.38 | 10.98 | 27.98 | 43.53 |
|  ICD-10-CM  |  Head  | 90.89 | 14.37 | 22.67 | 24.29 | 26.11 | 53.82 | 69.16 |
|  ICD-10-CM  | Medium | 91.67 | 9.56 | 18.32 | 20.77 | 25.68 | 52.85 | 66.63 |
|  ICD-10-CM  |  Tail  | 86.55 | 4.81 | 8.69 | 12.74 |  8.86  | 22.75 | 37.97 |

ROC curves on Disorders and ICD-10-CM, including head/medium/tail parts respectively. The shadow in the figure shown the 95\% CI~(Confidence interval) and FM, KE are short for Fusion Module and Knowledge Enhancement.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/ROCs.png"/>

## Comparison

The AUC Score Comparison on Various External Datasets. SOTA denotes the best performance of former works (pointed
with corresponding reference) on the datasets. Scratch means use our model but train from scratch. Ours means use our checkpoint to fintune.

|   Dataset   | Scratch | Ours | SOTA |
| :---------: | :-----: | :---: | :---: |
| VinDr-Mammo |  76.25  | 78.53 | 77.50 |
|    CXR14    |  79.12  | 83.38 | 82.50 |
| VinDr-Spine |  87.35  | 87.73 | 88.90 |
| MosMedData |  71.24  | 75.39 | 68.47 |
|    ADNI    |  82.40  | 84.21 | 79.34 |

## Acknowledgment

We sincerely thank all the contributors who uploaded the relevant data in our dataset online. We appreciate their willingness to make these valuable cases publicly available.

## Contact

If you have any questions, please feel free to contact [three-world@sjtu.edu.cn](https://qiaoyu-zheng.github.io).
