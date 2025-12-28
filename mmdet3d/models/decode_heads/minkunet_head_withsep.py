# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class MinkUNetHeadWithSep(Base3DDecodeHead):
    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        self.cls_ids_for_sep = kwargs.get('cls_ids_for_sep', list(range(num_classes)))
        if 'cls_ids_for_sep' in kwargs:
            kwargs.pop('cls_ids_for_sep')
        super().__init__(channels, num_classes, **kwargs)
        self.sep_heads = nn.ModuleDict()
        for cls_id in self.cls_ids_for_sep:
            self.sep_heads[str(cls_id)] = nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Linear(channels, 1, bias=True) 
                )
        self.finetune_sep_heads = kwargs.get('finetune_sep_heads', True)

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Linear(channels, num_classes)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs)

    def predict(self, inputs: Tensor,
                batch_data_samples: SampleList,
                return_logits=False
                ) -> List[Tensor]:
        seg_logits = self.forward(inputs)
        batch_idx = torch.cat(
            [data_samples.batch_idx for data_samples in batch_data_samples])
        seg_logit_list = []
        for i, data_sample in enumerate(batch_data_samples):
            seg_logit = seg_logits[batch_idx == i]
            seg_logit = seg_logit[data_sample.point2voxel_map]
            seg_logit_list.append(seg_logit)
        return seg_logit_list

    def forward(self, x: Tensor) -> Tensor:
        return self.cls_seg(x)
    
    def loss(self, 
             inputs: dict, 
             batch_data_samples: SampleList,
             train_cfg=None, 
             return_logits=False):
        
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        sep_logits = []
        for i in range(self.num_classes):
            if i in self.cls_ids_for_sep:
                sep_logits.append(self.sep_heads[str(i)](inputs))
            else:
                sep_logits.append(None)
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss_sep = self.loss_sep_logits(sep_logits, seg_label)
        if self.finetune_sep_heads:
            losses = {'loss_sep': loss_sep}

        if return_logits:
            return losses, seg_logits
        else:
            return losses
        

    def loss_sep_logits(
        self,
        sep_logits: List[torch.Tensor],   # len = C；每个形状 [N,1] 或 [N]
        gt_label: torch.Tensor,           # [N]，Long，取值 0..C-1；0=background
        *,
        bg_idx: int = 0,
        class_weights: Optional[torch.Tensor] = None,  # [C]，可选的类别级权重
        weight_bg: float = 0.5,           # 若没提供 class_weights，则对背景减权
        use_pos_weight: bool = True,      # 为每类用 neg/pos 做 pos_weight
        focal_gamma: Optional[float] = None,  # 设为 2.0 可用 Focal-BCE
        ignore_index: Optional[int] = None,   # e.g., -1：忽略这些点
        reduction: str = "mean",          # 'mean' or 'sum'
        aggregate: str = "mean_over_classes"  # 或 'mean_over_points_all'
    ) -> torch.Tensor:
        """
        返回标量 loss（保持可微）。
        """
        C = len(sep_logits)
        N = gt_label.numel()
        device = gt_label.device

        y = gt_label.to(device).long()
        if ignore_index is not None:
            valid_mask = (y != ignore_index)
        else:
            valid_mask = torch.ones_like(y, dtype=torch.bool)

        # 类别级权重
        if class_weights is not None:
            w_cls = class_weights.to(device).float()
            assert w_cls.numel() == C
        else:
            w_cls = torch.ones(C, device=device)
            if 0 <= bg_idx < C:
                w_cls[bg_idx] = float(weight_bg)

        losses_per_class = []

        for c in range(C):
            if sep_logits[c] is None:
                continue
            logit_c = sep_logits[c].to(device).view(-1)  # [N]
            assert logit_c.shape[0] == N, f"class {c} 的 N 不一致"

            # 仅对有效样本计损失
            logit_c = logit_c[valid_mask]
            y_valid = y[valid_mask]

            # 二元目标：是否属于第 c 类
            target = (y_valid == c).float()

            # BCE（带可选 pos_weight）
            if use_pos_weight:
                pos = target.sum()
                neg = target.numel() - pos
                # 当本 batch 无正样本时，pos_weight 的值无效，但仍给个有限值
                pos_weight = (neg / pos.clamp_min(1)).to(device)
            else:
                pos_weight = None

            ce = F.binary_cross_entropy_with_logits(
                logit_c, target, pos_weight=pos_weight, reduction="none"
            )

            if focal_gamma is not None:
                p = torch.sigmoid(logit_c)
                p_t = torch.where(target > 0.5, p, 1 - p)
                ce = (1 - p_t).pow(float(focal_gamma)) * ce  # Focal-BCE

            # 该类先做样本内聚合
            ce_c = ce.mean() if reduction == "mean" else ce.sum()
            # 类别级权重
            ce_c = ce_c * w_cls[c]

            losses_per_class.append(ce_c)

        if aggregate == "mean_over_classes":
            loss = torch.stack(losses_per_class).mean() if reduction == "mean" else torch.stack(losses_per_class).sum()
        elif aggregate == "mean_over_points_all":
            # 把每类损失按类权重求和后再 /C，近似所有点上的平均（各类贡献随样本量变化）
            loss = torch.stack(losses_per_class).sum() / C if reduction == "mean" else torch.stack(losses_per_class).sum()
        else:
            raise ValueError("aggregate 只能是 'mean_over_classes' 或 'mean_over_points_all'")

        return loss