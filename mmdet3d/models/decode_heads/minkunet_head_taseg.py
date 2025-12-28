# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class MinkUNetHeadTASeg(Base3DDecodeHead):
    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(channels, num_classes, **kwargs)
        self.classifier_fusion = nn.Sequential(
            nn.Linear(channels*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

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
        seg_logits = self.classifier_fusion(inputs['x_fused'])
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
        
        seg_logits = self.forward(inputs['x'])
        seg_logits_fusion = self.classifier_fusion(inputs['x_fused'])
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        losses_fusion = self.loss_by_feat(seg_logits_fusion, batch_data_samples)
        losses['loss_sem_seg'] += losses_fusion['loss_sem_seg']

        if return_logits:
            return losses, seg_logits
        else:
            return losses