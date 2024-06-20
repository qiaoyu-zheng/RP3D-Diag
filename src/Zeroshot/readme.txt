Codes in this folder can be used to conduct zeroshot experiment.

Here we provide the most directly situation, that is, pick one class in RP3D-DiagDS to map the class in external dataset.

Use the same model checkpoint as fine-tuning for zeroshot experiments, because the external datasets are all image-level. 
First run predict.sh to get the inference results on the external dataset, and then run eval.py to get the evaluation results.
The checkpoint can be downloaded from https://huggingface.co/QiaoyuZheng/RadDiag/tree/main/RP3D_5569_res_ART_256_32_BCE_T_2048_L_F_F_F_N