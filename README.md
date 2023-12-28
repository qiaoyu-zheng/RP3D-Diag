# RP3D-DiagModel
The official codes for paper "Large-scale Long-tailed Disease Diagnosis on Radiology"

[ArXiv Version](https://arxiv.org/abs/2312.16151)

In this paper, we build up an academically accessible, large-scale diagnostic dataset, present a knowledge enhanced model architecture that enables to process arbitrary number of input scans from various imaging modalities, and initialize a new benchmark for multi-modal multi-anatomy long-tailed diagnosis. Our method shows superior results on it. Additionally, our final model serves as a pre-trained model, and can be finetuned to benefit diagnosis on various external datasets.
## Dataset
Overview of RP3D-DiagDS. There are **39,026 cases (192,675 scans)** across 7 human anatomy regions and 9 diverse modalities covering **930 ICD-10-CM codes**.

<img src="https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/Images/RP3D-DiagDS.png"/>

The images used in our dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/1E_uSoCLm5H66a7KkpRfi1g?pwd=urfg)

The train/test split strategy, label csv files and model checkpoints can be found in [HuggingFace](https://huggingface.co/datasets/QiaoyuZheng/RP3D-DiagDS).

## Model Checkpoint
The model checkpoint can be found in [HuggingFace] (Uploading).

It can be easily loaded through:
```
Model
```
## Training

## Evaluation

## Acknowledgment
We sincerely thank all the contributors who uploaded the relevant data in our dataset online. We appreciate their willingness to make these valuable cases publicly available.

## Contact
If you have any questions, please feel free to contact.
