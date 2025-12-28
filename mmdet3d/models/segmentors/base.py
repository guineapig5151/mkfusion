# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

from mmengine.model import BaseModel
import torch
from torch import Tensor

from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils import OptConfigType, OptMultiConfig


class Base3DSegmentor(BaseModel, metaclass=ABCMeta):
    """Base class for 3D segmentors.

    Args:
        data_preprocessor (dict or ConfigDict, optional): Model preprocessing
            config for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``, ``mean`` and
            ``std``. Defaults to None.
       init_cfg (dict or ConfigDict, optional): The config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(Base3DSegmentor, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: Whether the segmentor has neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        """bool: Whether the segmentor has auxiliary head."""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self) -> bool:
        """bool: Whether the segmentor has decode head."""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @property
    def with_regularization_loss(self) -> bool:
        """bool: Whether the segmentor has regularization loss for weight."""
        return hasattr(self, 'loss_regularization') and \
            self.loss_regularization is not None

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList) -> Tensor:
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict or List[dict]): Input sample dict which includes
                'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor): Image tensor has shape (B, C, H, W).
            data_samples (List[:obj:`Det3DDataSample`], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            # self.eval()
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # import torch.nn as nn
            # for m in self.modules():
            #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.train()
            #     elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)): m.eval()
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 batch_inputs: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def postprocess_result(self, seg_logits_list: List[Tensor],
                           batch_data_samples: SampleList, points = None) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_logits_list (List[Tensor]): List of segmentation results,
                seg_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """


        for i in range(len(seg_logits_list)):
            seg_logits = seg_logits_list[i]
            seg_pred = seg_logits.argmax(dim=0)

            seg_pred_top2 = seg_logits.argsort(dim=0, descending=True)
            seg_prob_top2 = torch.softmax(seg_logits, dim=0).gather(dim=0, index=seg_pred_top2)[:2]
            seg_pred_top2 = seg_pred_top2[:2]

            save = False
            if save:
                save_path = 'vis_paper/mkfusion_tj4d/'
                gt_save_path = 'vis_paper/gt_tj4d/'
                import os
                os.makedirs(gt_save_path, exist_ok=True)
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                try:
                    name = batch_data_samples[0].img_path.split('/')[-1].replace('png', 'bin')
                except:
                    name = batch_data_samples[0].img_path[0].replace('png', 'bin')
                import numpy as np
                pts_np = points[0][:, :3].cpu().numpy().astype(np.float32)
                seg_np = seg_pred.cpu().numpy().astype(np.float32)
                seg_score = seg_logits.max(0)[0].sigmoid().cpu().numpy().astype(np.float32)

                # 合并为一个数组（可选，视具体结构需求而定）
                combined = np.concatenate((pts_np, seg_np.reshape(-1, 1), seg_score.reshape(-1, 1)), axis=1)  # 每行：x,y,z,label

                # 保存为 .bin 文件
                combined.tofile( save_path + name)

                combined_gt = np.concatenate((pts_np, batch_data_samples[0].eval_ann_info['pts_semantic_mask'].reshape(-1, 1).astype(np.float32), np.ones_like(pts_np[:, 0:1])), axis=1)
                combined_gt.tofile( gt_save_path + name)

            if points != None:
                batch_data_samples[i].set_data({
                'pts_seg_logits': PointData(**{'pts_seg_logits': seg_logits}),
                'pred_pts_seg': PointData(**{'pts_semantic_mask': seg_pred}),
                'points': points[i],
                'pred_pts_seg_top2': PointData(**{'pred_pts_seg_top2': seg_pred_top2}),
                'prob_pts_seg_top2': PointData(**{'prob_pts_seg_top2': seg_prob_top2}),
            })
            else:
                batch_data_samples[i].set_data({
                'pts_seg_logits': PointData(**{'pts_seg_logits': seg_logits}),
                'pred_pts_seg': PointData(**{'pts_semantic_mask': seg_pred}),
                'pred_pts_seg_top2': PointData(**{'pred_pts_seg_top2': seg_pred_top2}),
                'prob_pts_seg_top2': PointData(**{'prob_pts_seg_top2': seg_prob_top2}),
            })    
        return batch_data_samples
