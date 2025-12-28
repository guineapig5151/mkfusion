import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential, SparseConv3d, SubMConv3d
from mmdet3d.registry import MODELS

@MODELS.register_module()
class STConv_input_stacom(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = model_cfg.get('input_channels', 128)
        self.xy_ks = model_cfg.get('xy_ks', 3)
        self.z_ks = model_cfg.get('z_ks', 3)
        # xyz
        self.st_conv = SparseSequential(
            SubMConv3d(input_channels, input_channels, [self.xy_ks, self.xy_ks, self.z_ks], stride=1, padding=[self.xy_ks//2, self.xy_ks//2, self.z_ks // 2], bias=False, indice_key='st_conv'),
            nn.BatchNorm1d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.height_conv = SparseSequential(
            SubMConv3d(input_channels, input_channels, [self.xy_ks, self.xy_ks, self.z_ks], stride=1, padding=[self.xy_ks//2, self.xy_ks//2, self.z_ks // 2], bias=False, indice_key='st_conv_height'),
            nn.BatchNorm1d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 1, 2]] # xyz
        spatial_shape = x_conv.spatial_shape[:-1]

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

    def stconv2d(self, sparse_tensor, prev_feats):
        indices_prev = []
        indices_cur = sparse_tensor.indices
        indices_cur = torch.cat([indices_cur[:, 0:3], torch.zeros((indices_cur.shape[0], 1), dtype=indices_cur.dtype, device=indices_cur.device)], dim=1)
        for k, prev_feat in enumerate(prev_feats):
            indices = prev_feat.indices
            indices = torch.cat([indices[:, 0:3], (k + 1) + torch.zeros((indices.shape[0], 1), dtype=indices.dtype, device=indices.device)], dim=1)
            indices_prev.append(indices)

        indices_all = torch.cat([indices_cur, *indices_prev], dim=0)
        _, sorted_idxs = torch.sort(indices_all[:, 0])
        indices_all = indices_all[sorted_idxs]
        features_all = torch.cat([sparse_tensor.features, *[prev_feat.features for prev_feat in prev_feats]], dim=0)
        features_all = features_all[sorted_idxs]

        st_tensor = SparseConvTensor(
            features=features_all,
            indices=indices_all.int(),
            spatial_shape=[sparse_tensor.spatial_shape[0], sparse_tensor.spatial_shape[1], len(prev_feats) + 1],
            batch_size=sparse_tensor.batch_size
        )
        st_tensor = self.st_conv(st_tensor)
        st_tensor = self.bev_out(st_tensor)

        return st_tensor
    
    def hash_test(self, x, y):
        """检查 y.indices 是否是 x.indices 的子集，适用于任意维度 (n, d)"""
        if x is None or y is None:
            return None  # 或者返回 0，取决于你的应用场景

        if not hasattr(x, 'indices') or not hasattr(y, 'indices'):
            raise ValueError("输入数据必须包含 `indices` 属性")

        return_indices_cur = x.indices  # (n, d)
        mask_indices_cur = y.indices  # (m, d)

        if return_indices_cur.shape[1] != mask_indices_cur.shape[1]:
            raise ValueError("x 和 y 的 indices 维度必须相同")

        d = return_indices_cur.shape[1]  # 获取索引维度
        hash_tensor = torch.tensor([1025 ** i for i in range(d)]).cuda()  # 适配 n 维度

        # 计算哈希值
        hash_a = torch.sum(return_indices_cur * hash_tensor, dim=1)
        hash_b = torch.sum(mask_indices_cur * hash_tensor, dim=1)

        # 检查 hash_b 是否在 hash_a 中
        mask_test = torch.isin(hash_b, hash_a)
        indice_no = y.indices[~mask_test]
        if  torch.sum(~mask_test):
            print(indice_no)
        return torch.sum(~mask_test)  # 计算 y 中不在 x 中的元素数量

    
    def forward(self, x, before_feat_dict, cur_loc_name): # xyz
        prev_feats = []
        before_feat_list = before_feat_dict['before_feat_list']
        for i in range(len(before_feat_list)):
            prev_feats.append(before_feat_list[i][cur_loc_name])

        height = x.spatial_shape[-1]

        bev_out_list = []
        for height_id in range(height):
            heightmask = x.indices[:, -1] == height_id
            x_2d = SparseConvTensor(
                features=x.features[heightmask],
                indices=x.indices[heightmask][:, :-1],
                spatial_shape=x.spatial_shape[:-1],
                batch_size=x.batch_size
            )
            prev_feats_2d = []
            for tem_id, sparse_tensor in enumerate(prev_feats):
                heightmask = sparse_tensor.indices[:, -1] == height_id
                prev_feats_2d.append(SparseConvTensor(
                    features=sparse_tensor.features[heightmask],
                    indices=sparse_tensor.indices[heightmask][:, :-1],
                    spatial_shape=sparse_tensor.spatial_shape[:-1],
                    batch_size=sparse_tensor.batch_size
                ))

            bev_out = self.stconv2d(x_2d, prev_feats_2d)

            mask = self.hash_test(bev_out, x_2d) 
            assert mask == 0

            bev_out_list.append(bev_out)
        
        indices_height_list = []
        feat_list = []
        for height_id in range(height):
            indices_cur = bev_out_list[height_id].indices
            indices_height= torch.cat((indices_cur, height_id * torch.ones((indices_cur.shape[0], 1), dtype=indices_cur.dtype, device=indices_cur.device)), dim=1)
            indices_height_list.append(indices_height)
            feat_list.append(bev_out_list[height_id].features)

        feat = torch.cat(feat_list)
        indices = torch.cat(indices_height_list)
        sorted_tensor, sort_indices = torch.sort(indices[:, 0])

        sparse_tensor_rev = SparseConvTensor(
            features=feat[sort_indices],
            indices=indices[sort_indices],
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        mask = self.hash_test(sparse_tensor_rev, x)
        assert mask == 0

        sparse_tensor_rev = self.height_conv(sparse_tensor_rev)

        mask = self.hash_test(sparse_tensor_rev, x)
        assert mask == 0

        return sparse_tensor_rev  