# Instructions for RP3D_Demo

Here we give the instructions to run RP3D_Demo we provided from scratch.



## Example Data

Here we provide **3** cases formatted as .npy files in .../RP3D_Demo/DataPath/DemoData/

The data info file is .../RP3D_Demo/DataPath/demoData.json

`Case 1`: This case contains one CT volumn and one MRI volumn. For more intuitionistic information, please refer to [Radiopaedia Case 1](https://radiopaedia.org//cases/perforated-saccular-abdominal-aortic-aneurysm-1?lang=us)

`Case 2`: This case contains one X-ray image and two CT volumns. For more intuitionistic information, please refer to [Radiopaedia Case 2](https://radiopaedia.org//cases/hip-dislocation-inferior-1?lang=us)

`Case 3`: This case contains one X-rat image and two MRI volumns. For more intuitionistic information, please refer to [Radiopaedia Case3](https://radiopaedia.org//cases/segond-fracture-5?lang=us)



## Environment Setup

`Step1`: 

```cmd
pip install -r requirements
```

Please click [requirements.txt](https://github.com/qiaoyu-zheng/RP3D-Diag/blob/main/requirements.txt ) for detailed information

`Step2`:

Download the model checkpoint folder [RP3D_5569_res_ART_256_32_BCE_T_4_2_0.7_2048_E_F_F_F_N](https://huggingface.co/QiaoyuZheng/RP3D-DiagModel)

Then add this folder to the the Path: [.../RP3D_Demo/Logout/](https://github.com/qiaoyu-zheng/RP3D-Diag/tree/main/RP3D_Demo/Logout)



## Files Modification

Please modify all the relative paths in files to the absolute path in your machine. These files include:

`.../RP3D_Demo/Model/modelRadNet.py`: Change all the sys path to fit your environment.

`.../RP3D_Demo/Model/vit2D.py`: Change all the sys path to fit your environment.

`.../RP3D_Demo/Model/vitFuse.py`: Change all the sys path to fit your environment.

`.../RP3D_Demo/eval.py`: Change all the sys path to fit your environment.

`.../RP3D_Demo/eval.sh`: Change all the sys path to fit your environment.

`.../RP3D_Demo/verify.py`: Change all the sys path to fit your environment.



## Running Process

In the beginning , all the required files will form the structure in the following

.../RP3D_Demo

|-- DataOutput

|-- DataPath

​     |-- DemoData

​          |-- data

​	  |--key

​     |-- demoData.json

​     |-- disorder_label_dict.json

|-- Datset

​     |-- dataset.py

|-- Logout

​     |-- RP3D_5569_res_ART_256_32_BCE_T_4_2_0.7_2048_E_F_F_F_N

​     |-- log0Eval

|-- Loss

​     |-- AllLosses.py

|-- Model

​     |-- trainsformer_encoder

​     |-- modelRadnet.py

​     |-- position_encoding.py

​     |-- resAttention.py

​     |-- resnet2D.py

​     |-- vit2D.py

​     |-- vitFuse.py

|-- Utils

​     |-- utils.py

|-- eval.py

|-- eval.sh

|-- verify.py



In `eval.py`, since there is no training process, we can set train_path = eval_path = '.../RP3D_Demo/DataPath/demoData.json'

in `eval.sh`, keep the parameter **FUSE="early"**.

Then run in terminal:

```cmd
cd .../RP3D_Demo
bash eval.sh
```

After running, there will be two files generated, they are:

`.../RP3D_Demo/DataOutput/eval_logits.npy`: the model's prediction logits.

`.../RP3D_Demo/DataOutput/eval_labels.npy`: corresponding label.

Then run in terminal:

```cmd
python .../RP3D_Demo/verify.py
```

The final output is:

```
Case 0, gt disorders: ['Abdominal aortic aneurysm', 'Abdominal aortic aneurysm rupture'], predict result: [True, True], predict score: [0.6873902082443237, 0.5604543089866638], corresponding thresholds: [0.3722658157348633, 0.14874456822872162]

Case 1, gt disorders: ['Hip dislocation'], predict result: [True], predict score: [0.5709783434867859], corresponding thresholds: [0.2768756449222564]

Case 2, gt disorders: ['Anterior cruciate ligament tear', 'Segond fracture'], predict result: [True, True], predict score: [0.24337439239025116, 0.27158036828041077], corresponding thresholds: [0.094383917748928, 0.179479107260704]
```





