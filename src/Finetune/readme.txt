Codes in this folder can be used to finetune or train from scratch the model to fit other datasets.

In Dataset/dataset.py, some functions can be used to preprocess corresponding external dataset.
These datasets can be downloaded using urls as we provided in this project.

As these datasets are not formulated as case-level, so our early fusion strategy is not appropriate here
Please set FUSE="late" and KE=False to further tune or train from scratch using our model architecture.

The corresponding checkpoint can be found in our repository https://huggingface.co/QiaoyuZheng/RadDiag/tree/main/RP3D_5569_res_ART_256_32_BCE_T_4_2_0.7_2048_E_F_F_F_N.
