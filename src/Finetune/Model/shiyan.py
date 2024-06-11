# import torch 
# import torch.nn.functional as F


# mask = torch.randint(0,2,(2,4), dtype=torch.float32)
# print(mask)
# mask = F.pad(mask, (1, 0), "constant", 1)
# print(mask)


import torch

def generate_normalized_gaussian_labels(length=32, center=21, std_dev_fraction=1/8, cutoff_point=None):
    x = torch.arange(length).float()
    std_dev = length * std_dev_fraction
    labels = torch.exp(-0.5 * ((x - center) / std_dev) ** 2)
    
    # 如果指定了截断点，应用截断
    if cutoff_point is not None:
        labels[cutoff_point:] = 0
    
    # 归一化标签使其和为1
    labels /= labels.sum()
    
    return labels

# 示例：生成并归一化标签
length = 32
center = 21
std_dev_fraction = 1/16
cutoff_point = 25  # 可选的截断点
labels = generate_normalized_gaussian_labels(length, center, std_dev_fraction, cutoff_point)
print(labels)
# 验证标签和为1
print(f"Sum of labels: {labels.sum()}")