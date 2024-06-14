import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

thds = [[0.3722658157348633, 0.14874456822872162], [0.2768756449222564], [0.094383917748928, 0.179479107260704]]
with open("/home/qiaoyuzheng/MedVisionDemo/DataPath/disorder_label_dict.json", 'r') as f:
    disorder_label_dict = json.load(f)
disorders = np.array(list(disorder_label_dict.keys()))

logits = np.load("/home/qiaoyuzheng/MedVisionDemo/DataOutput/eval_logits.npy")
labels = np.load("/home/qiaoyuzheng/MedVisionDemo/DataOutput/eval_labels.npy")
scores = sigmoid(logits)

indices = []
names = []
for i in range(len(labels)):
    row_indices = [j for j, x in enumerate(labels[i]) if x == 1]
    indices.append(row_indices)
    names.append(disorders[row_indices].tolist())

predict_score = []
predict_res = []
predict_thds = []
for i in range(len(scores)):
    selected_scores = scores[i][indices[i]]
    selected_thds = np.array(thds[i])
    predict_thds.append(selected_thds.tolist())
    predict_score.append(selected_scores.tolist())
    predict_res.append((selected_scores>=selected_thds).tolist())

for i in range(len(predict_res)):
    print(f"Case {i}, gt disorders: {names[i]}, predict result: {predict_res[i]}, predict score: {predict_score[i]}, corresponding thresholds: {predict_thds[i]}")