from typing import Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS  # 或 from mmdet.registry import MODELS

@MODELS.register_module()
class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[Union[float, List[float]]] = None,
                 ignore_index: int = 255,
                 reduction: str = 'mean',
                 avg_non_ignore: bool = True,
                 loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.avg_non_ignore = avg_non_ignore
        self.loss_weight = loss_weight
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)  # 标量统一缩放
        else:
            self.alpha = None

    def forward(self,
                pred: torch.Tensor,     # [N, C] 或 [B, C, ...] -> 会自动展平到 [N, C]
                target: torch.Tensor,   # [N] 或 [B, ...]
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        self.ignore_index = kwargs.get('ignore_index', self.ignore_index)
        reduction = reduction_override if reduction_override else self.reduction

        # 展平到 [N, C], target [N]
        if pred.dim() > 2:
            C = pred.size(1)
            pred = pred.permute(0, *range(2, pred.dim()), 1).reshape(-1, C)
            target = target.reshape(-1)
            if weight is not None:
                weight = weight.reshape(-1)

        N, C = pred.shape
        device, dtype = pred.device, pred.dtype

        valid = target != self.ignore_index
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        pred = pred[valid]
        target = target[valid]
        if weight is not None:
            weight = weight[valid]

        logpt = F.log_softmax(pred, dim=1)
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)  # [M]
        pt = logpt_t.exp()
        focal = (1.0 - pt).clamp(min=0).pow(self.gamma)

        if self.alpha is not None:
            a = self.alpha.to(device=device, dtype=dtype)
            if a.numel() == 1:
                alpha_t = a.expand_as(pt)
            else:
                assert a.numel() == C, f'alpha length {a.numel()} != num_classes {C}'
                alpha_t = a[target]
            loss = -alpha_t * focal * logpt_t
        else:
            loss = -focal * logpt_t

        if weight is not None:
            loss = loss * weight

        if reduction == 'none':
            return self.loss_weight * loss
        if self.avg_non_ignore:
            denom = valid.sum().clamp(min=1).to(loss.dtype) if avg_factor is None else torch.tensor(float(avg_factor), device=device, dtype=loss.dtype)
            return self.loss_weight * (loss.sum() / denom)
        else:
            if reduction == 'mean':
                denom = (target.numel() if avg_factor is None else float(avg_factor))
                return self.loss_weight * (loss.sum() / denom)
            elif reduction == 'sum':
                return self.loss_weight * loss.sum()