import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

@MODELS.register_module()
class DiceLoss(nn.Module):
    def __init__(self,
                 smooth: float = 1.0,
                 ignore_index: int = 255,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        self.ignore_index = kwargs.get('ignore_index', self.ignore_index)
        reduction = reduction_override if reduction_override else self.reduction

        # pred -> [N, C]
        if pred.dim() > 2:
            C = pred.size(1)
            pred = pred.permute(0, *range(2, pred.dim()), 1).reshape(-1, C)
            target = target.reshape(-1)
            if weight is not None: weight = weight.reshape(-1)

        valid = target != self.ignore_index
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        pred = pred[valid]  # [M, C]
        target = target[valid]  # [M]
        prob = F.softmax(pred, dim=1)

        C = prob.size(1)
        onehot = F.one_hot(target, num_classes=C).to(prob.dtype)  # [M, C]

        inter = (prob * onehot).sum(dim=0)  # [C]
        union = prob.sum(dim=0) + onehot.sum(dim=0)  # [C]
        dice = (2 * inter + self.smooth) / (union + self.smooth)  # [C]
        loss = 1.0 - dice  # [C]

        if weight is not None:
            # 逐样本权重难以与类级 Dice 对齐；一般不使用
            pass

        if reduction == 'none':
            out = loss
        elif reduction == 'mean':
            out = loss.mean()
        else:
            out = loss.sum()
        return self.loss_weight * out