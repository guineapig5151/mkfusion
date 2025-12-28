import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential, SparseConv3d
from mmdet3d.registry import MODELS

@MODELS.register_module()
class Concat_input_stacom(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.downsample_times = 8
        self.voxel_size = 0.05
        self.point_cloud_range = torch.tensor([0, -25.6, -3, 51.2, 25.6, 2]).cuda()

        input_channels = model_cfg.get('input_channels', 128)
        self.xy_ks = model_cfg.get('xy_ks', 7)
        self.z_ks = model_cfg.get('z_ks', 3)
        # xyz
        self.st_conv = SparseSequential(
            SparseConv3d(input_channels, input_channels, [self.xy_ks, self.xy_ks, self.z_ks], stride=1, padding=[self.xy_ks//2, self.xy_ks//2, self.z_ks // 2], bias=False, indice_key='st_conv'),
            nn.BatchNorm1d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def bev_out_3d(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices # xyz
        spatial_shape = x_conv.spatial_shape

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, x, before_feat_dict, cur_loc_name): # xyz
        prev_feats = []
        before_feat_list = before_feat_dict['before_feat_list']

        indices_concat_list = []
        feat_concat_list = []
        indices_concat_list.append(x.indices)
        feat_concat_list.append(x.features)

        for i in range(len(before_feat_list)):
            cur_sp = before_feat_list[i][cur_loc_name]
            prev_feats.append(cur_sp)
            indices_concat_list.append(cur_sp.indices)
            feat_concat_list.append(cur_sp.features)

        indice_concat = torch.cat(indices_concat_list, dim=0)
        feat_concat = torch.cat(feat_concat_list, dim=0)
        x_out = SparseConvTensor(
            features=feat_concat,
            indices=indice_concat,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        x_out = self.bev_out_3d(x_out)
        return x_out