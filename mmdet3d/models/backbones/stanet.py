# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from .base_pointnet import BasePointNet
from ..layers.pointnet_modules.stanet_sa_module import STANetSAModule

ThreeTupleIntType = Tuple[Tuple[Tuple[int, int, int]]]
TwoTupleIntType = Tuple[Tuple[int, int, int]]
TwoTupleStrType = Tuple[Tuple[str]]


@MODELS.register_module()
class STANet(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods Sequence[Tuple[str]]: Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels: int,
                 num_samples: Tuple[int] = (2048, 1024, 512, 256),
                 num_neighbors: Tuple[int] = (30, 15),
                 tf_input_dims: Tuple[int] = (64, 192),
                 tf_hidden_sizes: Tuple[int] = (64, 256),
                 fps_mods: TwoTupleStrType = (('D-FPS'), ('FS'), ('F-FPS',
                                                                  'D-FPS')),
                 fps_sample_range_lists: TwoTupleIntType = ((-1), (-1), (512,
                                                                         -1)),
                 init_cfg: OptConfigType = None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.SA_modules = nn.ModuleList()
        self.rfe = nn.Sequential(
                nn.Linear(in_channels,64),
                nn.LeakyReLU(),
            )
        self.num_sa = len(num_neighbors)
        for sa_index in range(len(num_neighbors)):

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                STANetSAModule(
                    num_point=num_samples[sa_index],
                    n_neighbor=num_neighbors[sa_index],
                    tf_input_dim=tf_input_dims[sa_index],
                    tf_hidden_size=tf_hidden_sizes[sa_index],
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    # ra_norm_cfg is optional; default None keeps behavior unchanged
                    ra_norm_cfg=kwargs.get('ra_norm_cfg', None),
                    dataset=kwargs.get('dataset','VoD')
                    ))

    def measure_neighbors(self,high_xyz,low_xyz):
        """
        Args:
            high_xyz: B, M, 7
            low_xyz: B, N, 7
        Return:
            dist_map: B N M
        """
        chosed_info = [0,1]
        
        high_xyz = high_xyz[:,:,:3] # B,N,2
        low_xyz = low_xyz[:,:,:3] # B,M,2
        
        dist_map = torch.sum((low_xyz.unsqueeze(2) - high_xyz.unsqueeze(1))**2,dim=-1)

        return torch.sqrt(dist_map)

    def forward(self, points: Tensor):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the
                    input points.
        """
        # xyz, features = self._split_point_feats(points)
        xyz = points
        features = self.rfe(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [indices]
        out_sa_cls_features = []
        out_indices_rel = []

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_cls_features, cur_indices = self.SA_modules[i](
                i, sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
            out_indices_rel.append(cur_indices)
            out_sa_xyz.append(sa_xyz[-1])
            out_sa_features.append(sa_features[-1])
            out_sa_indices.append(sa_indices[-1])
            out_sa_cls_features.append(cur_cls_features)

        
        atten_l0_to_l2 = self.measure_neighbors(xyz,out_sa_xyz[-1]).permute(0,2,1)
        
        res = dict()
        res['l1_xy_idx'] = out_indices_rel[-2]
        res['l2_xy_idx'] = out_indices_rel[-1]
        res['radar_feat'] = features
        res['l1_cls_feat'] = out_sa_cls_features[-2]
        res['l2_cls_feat'] = out_sa_cls_features[-1] 
        res['l2_fusion_feat'] = out_sa_features[-1]
        res['l0_to_l1'] = self.measure_neighbors(xyz,out_sa_xyz[-2])
        res['l1_to_l2'] = self.measure_neighbors(out_sa_xyz[-2],out_sa_xyz[-1])
        res['l0_to_l2'] = atten_l0_to_l2

        return res
