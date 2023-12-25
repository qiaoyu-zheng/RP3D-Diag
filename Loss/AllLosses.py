import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

if __name__ == "__main__":
    lossfunc1 = MultiSoftLoss()
    # lossfunc2 = nn.BCEWithLogitsLoss()
    lossfunc3 = MultiLabelLoss(num_cls=3)
    lossfunc4 = BCEFocalLoss(num_cls=3)
    lossfunc5 = AsymmetricLoss()
    inputs = torch.tensor([[0.1,-0.3,-5],[0.6, 1.5, -0.4]],dtype=torch.float32)
    targets = torch.tensor([[1,0,0],[1,1,0]],dtype=torch.float32)
    print(lossfunc1(inputs, targets))
    # print(lossfunc2(inputs, targets))
    print(lossfunc3(inputs, targets))
    print(lossfunc4(inputs, targets))
    print(lossfunc5(inputs, targets))

