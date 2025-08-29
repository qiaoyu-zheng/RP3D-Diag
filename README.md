# RP3D-Diag

[Nature Communications](https://www.nature.com/articles/s41467-024-54424-6)  | The official codes implementation for paper "Large-scale Long-tailed Disease Diagnosis on Radiology Images"

[ArXiv Version](https://arxiv.org/abs/2312.16151)

In this paper, we build up an academically accessible, large-scale diagnostic dataset, present a knowledge enhanced model architecture that enables to process arbitrary number of input scans from various imaging modalities, and initialize a new benchmark for multi-modal multi-anatomy long-tailed diagnosis. Our method shows superior results on it. Additionally, our final model serves as a pre-trained model, and can be finetuned to benefit diagnosis on various external datasets.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=qiaoyu-zheng/RP3D-Diag&type=Date)](https://www.star-history.com/#qiaoyu-zheng/RP3D-Diag&Date)

## Dataset

**Overview of RP3D-DiagDS.** There are **40,936 cases (195,010 scans)** across 7 human anatomy regions and 9 diverse modalities covering **5569 disorders** mapped into **931 ICD-10-CM codes**.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/RP3D-DiagDS.png"/>



<!-- The train/test split strategy and label csv files can be found in HuggingFace repository [RP3D-DiagDS](https://huggingface.co/datasets/QiaoyuZheng/RP3D-DiagDS). (Under updating). -->

## Model

The overview of our model **RadDiag**. Three parts demonstrate our proposed visual encoders and fusion module, together
with the knowledge enhancement strategy respectively. a, The three types of vision encoder, i.e., ResNet-based, ViT-based, and
ResNet-ViT-mixing. b, The architecture of the fusion module. The figure shows the transformer-based fusion module, enabling
case-level information fusion. c, The knowledge enhancement strategy. We first pre-train a text encoder with extra medical
knowledge with contrastive learning, leveraging synonyms, descriptions, and hierarchy. Then we view the text embedding as a
natural classifier to guide the diagnosis classification.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/RP3D-DiagModel.png"/> 

The model checkpoint can also be found in HuggingFace repository [RadDiag](https://huggingface.co/QiaoyuZheng/RadDiag).

## Setup

### Environment

To Install the python environments:
```
pip install -r requirements.txt
```

### Data Preparation

* Download files from HuggingFace.
* Modify all the paths (files in DataPath, all the python files and bash files) to your own.
* The DataPath folder should be like:

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

## Quickstart Demo

**Here we use a demo to show in detail the directory structure and how to run the model for inference, as an example for reproducing the subsequent experiments.** 



To run this demo, please refer to [RP3D_Demo_Instruction.md](https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/RP3D_Demo_Instruction.md)




## Eval

1. cd .../src/Eval
2. add data file to ./DataPath
3. replace the relative path in ./Model/ and ./eval.py, ./eval.sh, etc with your absolute path.
4. set checkpoint in ./eval.sh
5. Run in terminal:  `bash eval.sh`

**For more info, please refer to readme files in src/Eval/**

## Train

### From Scratch

1. cd .../src/Train
2. add data file to ./DataPath/
3. replace the relative path in ./Model/ and ./train.py, ./train.sh, etc with your absolute path.
4. Run in terminal: `bash train.sh`


### Load Checkpoint

1. cd .../src/Train
2. add data file to ./DataPath/
3. replace the relative path in ./Model/ and ./train.py, ./train.sh, etc with your absolute path.
4. set checkpoint in ./train.sh
5. Run in terminal: `bash train.sh`

**For more info, please refer to readme files in src/Train/**

## Finetune

1. cd .../src/Finetune
2. add data file to ./DataPath/
3. replace the relative path in ./Model/ and ./train.py, ./train.sh, etc with your absolute path.
4. set checkpoint in ./train.sh
5. Run in terminal: `bash train.sh`

**For more info, please refer to readme files in src/Finetune/**

## Zeroshot

1. cd .../src/Zeroshot
2. add data file to ./DataPath/
3. replace the relative path in ./Model/ and ./predict.py, ./predict.sh, eval.py, etc with your absolute path.
4. set checkpoint in ./eval.sh
5. run in terminal: `bash predict.sh`
6. run in terminal: python eval.py

**For more info, please refer to readme files in src/Zeroshot/**


## Benchmark

**Classification results on Disorders and ICD-10-CM levels.** In this table **Fusion Module** and **Knowledge Enhancement** strategy are both used. We report the results on Head/Medium/Tail class sets separately.

| Granularity | Class |  AUC  |  AP  |  F1  |  MCC  | R@0.01 | R@0.05 | R@0.1 |
| :---------: | :----: | :---: | :---: | :---: | :---: | :----: | :----: | :---: |
|  Disorders  |  Head  | 94.41 | 20.27 | 30.21 | 32.27 | 41.93 | 71.09 | 81.15 |
|  Disorders  | Medium | 95.14 | 15.95 | 25.82 | 28.84 | 42.49 | 68.73 | 79.39 |
|  Disorders  |  Tail  | 90.96 | 7.75 | 12.68 | 17.49 | 13.13 | 28.88 | 44.08 |
|  ICD-10-CM  |  Head  | 91.27 | 14.59 | 22.81 | 25.12 | 27.87 | 57.75 | 72.32 |
|  ICD-10-CM  | Medium | 92.01 | 10.34 | 19.08 | 22.16 | 28.81 | 55.55 | 69.86 |
|  ICD-10-CM  |  Tail  | 88.11 | 5.57 | 10.48 | 14.68 |  12.75  | 30.11 | 51.77 |

ROC curves on Disorders and ICD-10-CM, including head/medium/tail parts respectively. The shadow in the figure shows the 95\% CI~(Confidence interval) and FM, KE are short for Fusion Module and Knowledge Enhancement.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/ROCs.png"/>

## Comparison

The AUC Score Comparison on Various External Datasets. SOTA denotes the best performance of former works (pointed
with the corresponding reference) on the datasets. Scratch means using our model but training from scratch. Ours means to use our checkpoint to fintune.

|   Dataset   | Scratch | Ours | SOTA |
| :---------: | :-----: | :---: | :---: |
| VinDr-Mammo |  76.25  | 78.46 | 77.50 |
|    CXR14    |  79.12  | 83.44 | 82.50 |
| VinDr-Spine |  87.35  | 87.92 | 88.90 |
| MosMedData |  72.36  | 76.79 | 68.47 |
|    ADNI    |  83.44  | 85.61 | 79.34 |
| NSCLC | 67.25 | 72.54 | N/A |
| TCGA | 88.66 | 95.17 | N/A |
| ISPY1 | 65.88 | 69.43 | N/A |
| ChestX-Det10 | 74.11 | 79.44 | N/A |
| CheXpert | 89.52 | 91.27 | 93.00 |
| COVID-19-Radio | 95.39 | 98.56 | N/A |
| IU-Xray | 74.01 | 76.04 | N/A |
| LNDb | 68.76 | 70.58 | N/A |
| PadChest | 74.22 | 76.15 | 77.30 |
| CC-CCII | 98.27 | 99.46 | 97.41 |
| RadChest | 74.22 | 76.15 | 77.30 |
| Brain-Tumor | 91.05 | 93.21 | N/A |
| Brain-Tumor-17 | 93.66 | 94.43 | N/A |
| POCUS | 94.31 | 95.46 | 94.00 |
| MURA | 86.25 | 88.31 | 92.90 |
| KneeMRI | 73.05 | 73.22 | N/A |
| CT-Kidney | 91.41 | 93.26 | N/A |

In zero-shot assessment, we evaluate the transferring ability of our final model on normal/abnormal diagnosis for the external dataset.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/NormalVisulization_adjust.png"/>

## Acknowledgment

We sincerely thank all the contributors who uploaded the relevant data in our dataset online. We appreciate their willingness to make these valuable cases publicly available.

## Citation
```
@article{zheng2024large,
  title={Large-scale long-tailed disease diagnosis on radiology images},
  author={Zheng, Qiaoyu and Zhao, Weike and Wu, Chaoyi and Zhang, Xiaoman and Dai, Lisong and Guan, Hengyu and Li, Yuehua and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={10147},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
