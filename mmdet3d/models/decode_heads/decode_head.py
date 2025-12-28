# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List

import torch
from mmengine.model import BaseModule, normal_init
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig


class Base3DDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.5.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_decode (dict or :obj:`ConfigDict`): Config of decode loss.
            Defaults to dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            class_weight=None, loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 1.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 255.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0.5,
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 loss_decode: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 conv_seg_kernel_size: int = 1,
                 ignore_index: int = 255,
                 init_cfg: OptMultiConfig = None) -> None:
        super(Base3DDecodeHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        else:
            self.loss_decode = nn.ModuleList()
            for loss_cfg in loss_decode:
                self.loss_decode.append(MODELS.build(loss_cfg))
        self.ignore_index = ignore_index

        self.conv_seg = self.build_conv_seg(
            channels=channels,
            num_classes=num_classes,
            kernel_size=conv_seg_kernel_size)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def init_weights(self) -> None:
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.conv_seg, mean=0, std=0.01)

    @abstractmethod
    def forward(self, feats_dict: dict) -> Tensor:
        """Placeholder of forward function."""
        pass

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Conv1d(channels, num_classes, kernel_size=kernel_size)

    def cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg, use_seg_logits=False):
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        if use_seg_logits:
            return losses, seg_logits
        else:
            return losses

    def predict(self, inputs: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        seg_logits = self.forward(inputs)

        return seg_logits

    def padding_concat(self, point_list):
        # 找到 point_list 中最大点数
        max_len = max(points.shape[0] for points in point_list)
        feature_dim = 1

        # 初始化一个全零数组
        padded_points = torch.zeros((len(point_list), max_len, feature_dim), dtype=point_list[0].dtype, device=point_list[0].device)

        # 遍历 point_list，填充到 padded_points
        for i, points in enumerate(point_list):
            padded_points[i, :points.shape[0], :] = points.reshape(-1, 1)

        return padded_points

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        return self.padding_concat(gt_semantic_segs).squeeze(-1)
        # return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logit: Tensor,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        if torch.isnan(seg_logit).any() or torch.isinf(seg_logit).any():
            print(torch.isnan(seg_logit).any())  # True 表示含 NaN
            print(torch.isinf(seg_logit).any())  # True 表示含 Inf

        if isinstance(self.loss_decode, nn.ModuleList):
            loss['loss_sem_seg'] = 0
            for loss_decode in self.loss_decode:
                loss['loss_sem_seg'] += loss_decode(
                    seg_logit, seg_label, ignore_index=self.ignore_index)
        else:
            loss['loss_sem_seg'] = self.loss_decode(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        
        if torch.isnan(loss['loss_sem_seg']).any():
            print('w')
        return loss

    def get_seg_logits(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg: ConfigType):
        seg_logits = self.forward(inputs)
        return seg_logits