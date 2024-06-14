import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class InfoNCELoss(nn.Module):
    def __init__(self, length=32, temperature=0.07, thd=5):
        super().__init__()
        self.temperature = temperature
        self.length = length
        self.thd = thd
    
    def generate_labels_mask_with_cutoff(self, centers, cutoffs, device):
        batch_size = centers.size(0)
        mask = torch.zeros((batch_size, self.length), device=device, dtype=torch.float32)
        indices = torch.arange(self.length).unsqueeze(0).repeat(batch_size, 1).to(device)
        anchor_masks = (indices >= (centers - self.thd).unsqueeze(1)) & (indices <= (centers + self.thd).unsqueeze(1))
        cut_masks = indices < cutoffs.unsqueeze(1)
        mask[anchor_masks & cut_masks] = 1
        # mask = F.pad(mask, (1, 0), "constant", 1)
        return mask

    def forward(self, pred, centers, cutoffs):
        """
        pred: shape=(batch_size, length)
        centers: shape=(batch_size,)
        cutoffs: shape=(batch_size,)
        """
        device = pred.device
        pos_mask = self.generate_labels_mask_with_cutoff(centers, cutoffs, device)
        neg_mask = 1 - pos_mask
        pred_exp = torch.exp(pred)
        pos_weight = torch.sum(pred_exp * pos_mask, dim=1, keepdim=True)    
        neg_weight = torch.sum(pred_exp * neg_mask, dim=1, keepdim=True)
        scores = pos_weight / (pos_weight + neg_weight)
        return -torch.mean(torch.log(scores))

        

class SoftCrossEntropy(nn.Module):
    def __init__(self, length, std_dev_fraction):
        super().__init__()
        self.length = length
        self.std_dev_fraction = std_dev_fraction

    def generate_gaussian_labels_with_cutoff(self, centers, cutoffs, device):
        batch_size = centers.size(0)
        x = torch.arange(self.length, device=device).repeat(batch_size, 1).float()
        
        # 计算每个样本的标签
        std_dev = self.length * self.std_dev_fraction
        labels = torch.exp(-0.5 * ((x - centers.unsqueeze(1)) / std_dev) ** 2)
        
        # 应用截断
        cutoffs_expanded = cutoffs.unsqueeze(1).expand(-1, self.length)
        labels = torch.where(x < cutoffs_expanded, labels, torch.zeros_like(labels))
        
        # 归一化标签使其总和为1，以保持整体面积为1
        labels_sum = labels.sum(dim=1, keepdim=True)
        labels /= labels_sum
        
        return labels
    
    def forward(self, pred, centers, cutoffs):
        """
        centers: shape=(batch_size,)
        cutoffs: shape=(batch_size,)
        """
        device = pred.device
        labels = self.generate_gaussian_labels_with_cutoff(centers, cutoffs, device)
        labels = labels.clamp(min=1e-5)
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=1e-5)
        # return -torch.sum(labels * torch.log(pred), dim=1).mean()
        return -torch.sum(labels * torch.log(pred), dim=1).mean() 


class KLwithGaussian(nn.Module):
    def __init__(self, length, std_dev_fraction):
        super().__init__()
        self.length = length
        self.std_dev_fraction = std_dev_fraction
        # self.loss = nn.KLDivLoss(reduction='batchmean')

    def generate_gaussian_labels_with_cutoff(self, centers, cutoffs, device):
        batch_size = centers.size(0)
        x = torch.arange(self.length, device=device).repeat(batch_size, 1).float()
        
        # 计算每个样本的标签
        std_dev = self.length * self.std_dev_fraction
        labels = torch.exp(-0.5 * ((x - centers.unsqueeze(1)) / std_dev) ** 2)
        
        # 应用截断
        cutoffs_expanded = cutoffs.unsqueeze(1).expand(-1, self.length)
        labels = torch.where(x < cutoffs_expanded, labels, torch.zeros_like(labels))
        
        # 归一化标签使其总和为1，以保持整体面积为1
        labels_sum = labels.sum(dim=1, keepdim=True)
        labels /= labels_sum
        
        return labels

    def forward(self, pred, centers, cutoffs):
        """
        centers: shape=(batch_size,)
        cutoffs: shape=(batch_size,)
        """
        device = pred.device
        labels = self.generate_gaussian_labels_with_cutoff(centers, cutoffs, device)
        labels = labels.clamp(min=1e-5)
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=1e-5)
        # assert torch.all(labels.sum(dim=1) - 1 < 1e-5)
        # assert torch.all(pred.sum(dim=1) - 1 < 1e-5)
        # print(pred.sum(dim=1))
        # log_pred = F.log_softmax(pred, dim=1)

        return (F.kl_div(pred.log(), labels, reduction='batchmean') + F.kl_div(labels.log(), pred, reduction='batchmean')) / 2

class MSEwithGaussian(nn.Module):
    def __init__(self, length, std_dev_fraction):
        super().__init__()
        self.length = length
        self.std_dev_fraction = std_dev_fraction

    def generate_gaussian_labels_with_cutoff(self, centers, cutoffs, device):
        batch_size = centers.size(0)
        x = torch.arange(self.length, device=device).repeat(batch_size, 1).float()
        
        # 计算每个样本的标签
        std_dev = self.length * self.std_dev_fraction
        labels = torch.exp(-0.5 * ((x - centers.unsqueeze(1)) / std_dev) ** 2)
        
        # 应用截断
        cutoffs_expanded = cutoffs.unsqueeze(1).expand(-1, self.length)
        labels = torch.where(x < cutoffs_expanded, labels, torch.zeros_like(labels))
        
        # 归一化标签使其总和为1，以保持整体面积为1
        labels_sum = labels.sum(dim=1, keepdim=True)
        labels /= labels_sum
        
        return labels

    def forward(self, pred, centers, cutoffs):
        """
        centers: shape=(batch_size,)
        cutoffs: shape=(batch_size,)
        """
        device = pred.device
        labels = self.generate_gaussian_labels_with_cutoff(centers, cutoffs, device)
        return F.mse_loss(pred, labels)



class CEFocalLoss(nn.Module):
    def __init__(self,  gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softm       
        if self.alpha:
            alpha = self.alpha[target]     
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class MultiLabelLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.pos_weights = torch.ones([num_cls]) * pos_weight
        
        # with open("/home/qiaoyuzheng/MedVision/DataPath/sorted_disease_cnt_dict.json", 'r') as f:
        #     disease_cnt = json.load(f)
        # samples_per_class = list(disease_cnt.values())[:85]
        # total_samples = sum(samples_per_class)
        # num_classes = len(samples_per_class)
        # weights = [total_samples / (num_classes * x) for x in samples_per_class]
        # weights_tensor = torch.tensor(weights, dtype=torch.float)
        # self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
        # self.bce_loss = nn.BCEWithLogitsLoss(weight=weights_tensor)
        self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, inputs, targets):
        return self.bce_loss(inputs.clamp(max=60,min=-60), targets)

class BCEFocalLoss(nn.Module):
    def __init__(self, num_cls=917, pos_weight=1, gamma=2.0, alpha=0.9, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weights = torch.ones([num_cls]) * pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights, reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.bce_loss(inputs.clamp(max=60,min=-60), targets)
        pt = torch.exp(logpt)
        # Compute the focal loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        # Apply weights alpha
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class MultiSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        Prob = torch.exp(inputs.clamp(max=60,min=-60))
        Prob_P = Prob * targets
        Prob_N = Prob * (1 - targets)
        Sum_N = torch.sum(Prob_N, dim=1, keepdim=True)
        Prob_PN = Prob_P + Sum_N
        score = Prob_P / Prob_PN
        mask = 1 - targets
        log_score = torch.log(score + mask)
        row_cnt = torch.sum(targets, dim=1, keepdim=True)
        loss_sample = - torch.sum(log_score, dim=1, keepdim=True) / row_cnt
        loss = torch.mean(loss_sample)
        return loss
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, use_weight = False):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if use_weight:
            return loss
        # return -loss.sum()
        return -torch.mean(loss)
    
class ClipLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        # norms = torch.sigmoid(inputs)
        return self.bce_loss(inputs.clamp(max=60,min=-60), targets)
    
class CombineLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs1, inputs2, inputs3, inputs4, targets1, targets2, targets3, targets4):
        loss1 = self.bce_loss(inputs1.clamp(max=60,min=-60), targets1)
        loss2 = self.bce_loss(inputs2.clamp(max=60,min=-60), targets2)
        loss3 = self.bce_loss(inputs3.clamp(max=60,min=-60), targets3)
        loss4 = self.bce_loss(inputs4.clamp(max=60,min=-60), targets4)
        return {
            "loss1": loss1,
            "loss2": loss2,
            "loss3": loss3,
            "loss4": loss4,
        }
    
class HieararchicalLoss(nn.Module):
    def __init__(self, map1to2, map2to3, map3to4, map2to1, map3to2, map4to3, alpha=0.5, beta=0.25):
        super().__init__()
        self.map1to2 = map1to2
        self.map2to3 = map2to3
        self.map3to4 = map3to4
        self.map2to1 = map2to1
        self.map3to2 = map3to2
        self.map4to3 = map4to3
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
    
    def forward(self, inputs1, inputs2, inputs3, inputs4, targets1, targets2, targets3, targets4):
        scores1, scores2, scores3, scores4 = torch.sigmoid(inputs1.clamp(max=60,min=-60)), torch.sigmoid(inputs2.clamp(max=60,min=-60)), torch.sigmoid(inputs3.clamp(max=60,min=-60)), torch.sigmoid(inputs4.clamp(max=60,min=-60))
        deduction2 = torch.matmul(scores1, self.map1to2)
        deduction3 = torch.matmul(scores2, self.map2to3)
        deduction4 = torch.matmul(scores3, self.map3to4)
        induction1 = torch.max(scores2.unsqueeze(-1).repeat(1, 1, self.map2to1.shape[1]) * self.map2to1, dim=1)[0]
        induction2 = torch.max(scores3.unsqueeze(-1).repeat(1, 1, self.map3to2.shape[1]) * self.map3to2, dim=1)[0]
        induction3 = torch.max(scores4.unsqueeze(-1).repeat(1, 1, self.map4to3.shape[1]) * self.map4to3, dim=1)[0]
        hi_scores1 = self.alpha * scores1 + (1 - self.alpha) * induction1
        hi_scores2 = self.alpha * scores1 + self.beta * deduction2 + (1 - self.alpha - self.beta) * induction2
        hi_scores3 = self.alpha * scores2 + self.beta * deduction3 + (1 - self.alpha - self.beta) * induction3
        hi_scores4 = self.alpha * scores3 + (1 - self.alpha) * deduction4
        loss1 = self.bce_loss(hi_scores1, targets1)
        loss2 = self.bce_loss(hi_scores2, targets2)
        loss3 = self.bce_loss(hi_scores3, targets3)
        loss4 = self.bce_loss(hi_scores4, targets4)
        return {
            "loss1": loss1,
            "loss2": loss2,
            "loss3": loss3,
            "loss4": loss4
        }


if __name__ == "__main__":
    # lossfunc1 = MultiSoftLoss()
    # # lossfunc2 = nn.BCEWithLogitsLoss()
    # lossfunc3 = MultiLabelLoss(num_cls=3)
    # lossfunc4 = BCEFocalLoss(num_cls=3)
    # lossfunc5 = AsymmetricLoss()
    # inputs = torch.tensor([[0.1,-0.3,-5],[0.6, 1.5, -0.4]],dtype=torch.float32)
    # targets = torch.tensor([[1,0,0],[1,1,0]],dtype=torch.float32)
    # print(lossfunc1(inputs, targets))
    # # print(lossfunc2(inputs, targets))
    # print(lossfunc3(inputs, targets))
    # print(lossfunc4(inputs, targets))
    # print(lossfunc5(inputs, targets))
    batch_size, length = 3, 32
    pred = torch.randn(batch_size, length+1)  # 随机生成预测值
    centers = torch.randint(0, length, (batch_size,))  # 随机选择中心
    cutoffs = torch.randint(0, length, (batch_size,))  # 确保cutoffs大于centers

    # 实例化并计算损失
    loss_module = InfoNCELoss(length=length)
    loss = loss_module(pred, centers, cutoffs)
    print(centers)
    print(cutoffs)
    print(loss)
