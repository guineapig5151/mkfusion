# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers.minkowski_engine_block import (
    IS_MINKOWSKI_ENGINE_AVAILABLE, MinkowskiBasicBlock, MinkowskiBottleneck,
    MinkowskiConvModule)
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig
from .unet3d_taseg import UNet3D

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    from torchsparse import PointTensor
    import torchsparse.nn.functional as F
    from .taseg_utils import voxel_to_point

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME
import copy
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
import numpy as np
from .voxel_fusion_utils import proj_to_img
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type


def check_nan(backend, x):
    return (backend == 'spconv' and torch.isnan(x.features).any()) or (backend == 'torchsparse' and torch.isnan(x.F).any())

@MODELS.register_module()
class MinkUNetBackboneTASeg(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        sparseconv_backend (str): Sparse convolutional backend.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 occ_loc = ['x_deconv4'],
                 occ_loss_loc = [],
                 voxel_size = None,
                 point_cloud_range = None,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backend: str = 'spconv',
                 spatial_shape = [1024, 1024, 41],
                 multiscale_out: bool = True,
                 with_image: bool = True,
                 init_cfg: OptMultiConfig = None, **kwargs) -> None:
        super().__init__(init_cfg)
        self.multiscale_out = multiscale_out
        self.coord_type = 'LIDAR'
        self.with_image = with_image
        if multiscale_out:
            assert sparseconv_backend == 'torchsparse', 'Currently, only torchsparse backend support multiscale out'
            if with_image:
                # 在 __init__ 直接初始化 FoV UNet3D
                # 约定：LiDAR 基础通道为 32，img_feats[0]/[2] 各 256 → FoV 输入维度 32+256+256=544
                self.if_dist = kwargs.get('if_dist', False)
                self.num_class_fov = kwargs.get('num_class_fov', kwargs.get('num_classes', 20))
                self.fov_in_dim = 512+in_channels
                self.lidar_backbone_fov: nn.Module = UNet3D(
                    input_dim=self.fov_in_dim,
                    num_class=self.num_class_fov,
                    if_dist=self.if_dist,
                )
        self.spatial_shape = torch.tensor(spatial_shape).cuda()
        
        self.occ_loc = occ_loc
        self.occ_loss_loc = occ_loss_loc
        if voxel_size is not None:
            self.voxel_size = torch.tensor(voxel_size).cuda()
            self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backend} not supported.'
        self.encoder_channels = encoder_channels
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        if sparseconv_backend == 'torchsparse':
            assert IS_TORCHSPARSE_AVAILABLE, \
                'Please follow `get_started.md` to install Torchsparse.`'
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' \
                else TorchSparseBottleneck
            # for torchsparse, residual branch will be implemented internally
            residual_branch = None
        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available,'
                              'turn to use spconv 1.x in mmcv.')
            input_conv = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' \
                else SparseBottleneck
            residual_branch = partial(
                make_sparse_convmodule,
                conv_type='SubMConv3d',
                order=('conv', 'norm'))
        elif sparseconv_backend == 'minkowski':
            assert IS_MINKOWSKI_ENGINE_AVAILABLE, \
                'Please follow `get_started.md` to install Minkowski Engine.`'
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            decoder_conv = partial(
                MinkowskiConvModule,
                conv_cfg=dict(type='MinkowskiConvNdTranspose'))
            residual_block = MinkowskiBasicBlock if block_type == 'basic' \
                else MinkowskiBottleneck
            residual_branch = partial(MinkowskiConvModule, act_cfg=None)

        self.conv_input = nn.Sequential(
            input_conv(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            input_conv(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    encoder_channels[i],
                    encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(
                                encoder_channels[i],
                                encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            decoder_layer = [
                decoder_conv(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                    indice_key=f'spconv{num_stages-i}')
            ]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] +
                                encoder_channels[-2 - i],
                                decoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i-1}'))
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i-1}'))
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))

    def indice_xyz_2_point_xyz(self, voxel_coords, downsample_times, zyx_input=False):
        assert voxel_coords.shape[-1] == 4
        voxel_centers = voxel_coords[:, 1:].float()  # (xyz)
        voxel_size = self.voxel_size * downsample_times
        pc_range = self.point_cloud_range[0:3].float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        voxel_centers = torch.cat((voxel_coords[:, 0:1], voxel_centers), dim=1)
        if downsample_times != 1:
            assert (torch.min(voxel_centers[:,1:],dim=0)[0] >= pc_range).all()
            assert (torch.max(voxel_centers[:,1:],dim=0)[0] <= self.point_cloud_range[3:]).all()
        return voxel_centers
    
    def create_occ_gt(self, x, gt_box_list):
        batch = len(gt_box_list)
        indices = x.indices
        downsample_times = self.spatial_shape[0] // x.spatial_shape[0]
        in_box_mask_list = []
        for batch_id in range(batch):
            cur_batch_gt_box = gt_box_list[batch_id]
            batch_mask = [indices[:, 0] == batch_id]
            cur_batch_indice = indices[batch_mask]
            indice_xyz = self.indice_xyz_2_point_xyz(cur_batch_indice, downsample_times)
            in_box_mask = points_in_boxes_part(indice_xyz[:,1:][None, :, :], cur_batch_gt_box.bboxes_3d.tensor[None, :, :])
            in_box_mask_list.append(in_box_mask.squeeze(0))
        return in_box_mask_list
    
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

    def indice_proj2img(self, non_empty_coor, batch_input_metas, downsample_times, zyx_input=False):
        centroids_img_list = []
        point_xyz_list = []
        bs = int(non_empty_coor[-1, 0]) + 1
        for i in range(bs):
            batch_mask = (non_empty_coor[:, 0] == i)
            voxel_center_grid = non_empty_coor[batch_mask] # voxel_center
            
            point_xyz = self.indice_xyz_2_point_xyz(voxel_center_grid, downsample_times, zyx_input)

            proj_mat = get_proj_mat_by_coord_type(batch_input_metas[i], self.coord_type)
            cur_batch_centroids_img = proj_to_img(point_xyz[:,1:], batch_input_metas[i], point_xyz.new_tensor(proj_mat), self.coord_type, normalize=True)
            centroids_img_list.append(cur_batch_centroids_img)
            point_xyz_list.append(point_xyz)
        return centroids_img_list, point_xyz_list


    def forward(self, 
                voxel_features: Tensor, 
                coors: Tensor, 
                img_feats, 
                batch_input_metas, 
                imgs,
                in_distill=False,
                gt_box_list = None,
                ):
        distill_return = {}

        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
            z = PointTensor(x.F, x.C.float())
            if self.with_image:
                coors_ts = x.C
                non_empty_coor_bfirst = torch.stack([
                    coors_ts[:, 3], coors_ts[:, 0], coors_ts[:, 1], coors_ts[:, 2]
                ], dim=1)

                # 利用 indice_proj2img（batch_first=True 假设）做投影，downsample_times=1
                centroids_img_list, _ = self.indice_proj2img(
                    non_empty_coor_bfirst, batch_input_metas, downsample_times=1, zyx_input=False
                )
                bs = int(coors_ts[:, 3].max().item()) + 1 if coors_ts.numel() > 0 else 0
                fov_global_idx = []
                img_feat_samples = []  # 为每个 FoV 点采样并拼接多尺度图像特征
                feats_for_sampling = [img_feats[0], img_feats[2]]
                for b in range(bs):
                    mask_ts_b = (coors_ts[:, 3] == b)
                    if not mask_ts_b.any():
                        continue
                    idx_b = torch.nonzero(mask_ts_b, as_tuple=False).view(-1)
                    uv_b = centroids_img_list[b]  # 形状 (N_b, 2)
                    # indice_proj2img 使用 normalize=True：uv 范围理论上在 [0,1] 内；
                    # 你的应用保证点都在图像内，仅对数值边界做 clamp。
                    u_norm = uv_b[:, 0]
                    v_norm = uv_b[:, 1]
                    u_norm = u_norm.clamp(0, 1)
                    v_norm = v_norm.clamp(0, 1)
                    # 记录该 batch 的全部非空体素索引（不再基于 in_img 过滤）
                    if idx_b.numel() == 0:
                        continue
                    fov_global_idx.append(idx_b)

                    if 0:
                        import os
                        from PIL import Image
                        import matplotlib.pyplot as plt
                        img_path = batch_input_metas[b]['img_path']
                        im = np.array(Image.open(img_path).convert('RGB'))
                        H0, W0 = im.shape[0], im.shape[1]
                        u_pix = (u_norm.clamp(0, 1) * (W0 - 1)).cpu().numpy()
                        v_pix = (v_norm.clamp(0, 1) * (H0 - 1)).cpu().numpy()
                        x = u_pix
                        y = v_pix
                        fig, ax = plt.subplots(figsize=(W0/200, H0/200), dpi=200)
                        ax.imshow(im, origin='upper')
                        # 放大可视化点的尺寸，使其更醒目（s 是点面积像素）
                        ax.scatter(x, y, s=36, c='r', linewidths=0, alpha=0.9)
                        ax.axis('off')
                        out_dir = batch_input_metas[b].get('debug_dir', 'debug_vis')
                        os.makedirs(out_dir, exist_ok=True)
                        base = os.path.basename(img_path)
                        out_path = os.path.join(out_dir, f'fov_overlay_{b}_{base}')
                        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                        

                    # 若传入了多尺度 img_feats（list of [B,C,H,W]），对每个尺度做最近邻采样并拼接
                    # 采样索引按各自尺度的 H_s/W_s 由归一化 uv 变换得到
                    feat_parts = []
                    for feat in feats_for_sampling:
                        _, C_s, H_s, W_s = feat.shape
                        # 将索引张量移动到与特征同一设备，避免 device mismatch
                        u_s = u_norm.to(feat.device)
                        v_s = v_norm.to(feat.device)
                        col = torch.clamp((u_s * (W_s - 1)).round().long(), 0, W_s - 1)
                        row = torch.clamp((v_s * (H_s - 1)).round().long(), 0, H_s - 1)
                        # 采样 [C_s, K] -> 转置为 [K, C_s]
                        feat_sample = feat[b, :, row, col].transpose(0, 1).contiguous()
                        feat_parts.append(feat_sample)
                    if len(feat_parts) > 0:
                        img_feat_samples.append(torch.cat(feat_parts, dim=1))

                if len(fov_global_idx) > 0:
                    fov_global_idx = torch.cat(fov_global_idx, dim=0)
                    coords_fov = coors_ts[fov_global_idx]
                    # LiDAR 基础通道（保守地全取当前特征，或按需要裁剪）
                    feats_lidar = x.F[fov_global_idx]
                    if len(img_feat_samples) > 0:
                        feats_img = torch.cat(img_feat_samples, dim=0)
                        # 与索引顺序一致：每个 batch 顺序附加
                        assert feats_img.size(0) == feats_lidar.size(0), 'FoV 图像特征与体素索引数量不一致'
                        feats_fov = torch.cat([feats_lidar, feats_img], dim=1)
                    else:
                        feats_fov = feats_lidar
                    x_fov_ms = torchsparse.SparseTensor(feats_fov, coords_fov)
                    distill_return['x_fov_ms'] = x_fov_ms
                else:
                    # 构造空张量（0×C 与 0×4 坐标），保持接口一致
                    feats_fov = x.F.new_zeros((0, x.F.shape[1]))
                    coords_fov = coors_ts.new_zeros((0, 4))
                    x_fov_ms = torchsparse.SparseTensor(feats_fov, coords_fov)
                    distill_return['x_fov_ms'] = x_fov_ms

        elif self.sparseconv_backend == 'spconv':
            spatial_shape_ = coors.max(0)[0][1:] + 1
            spatial_shape = self.spatial_shape
            # assert (spatial_shape >= spatial_shape_).all()
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)
        elif self.sparseconv_backend == 'minkowski':
            x = ME.SparseTensor(voxel_features, coors)

        x = self.conv_input(x)
        if self.multiscale_out: z0 = voxel_to_point(x, z, nearest=False)

        if check_nan(self.sparseconv_backend, x):
            print("NaN detected in x!")
            
        cur_loc_name = 'x_conv_input'
        if in_distill == True:
            distill_return[cur_loc_name] = x

        laterals = [x]
        for i, encoder_layer in enumerate(self.encoder):
            
            x = encoder_layer(x)

            if check_nan(self.sparseconv_backend, x):
                print("NaN detected in x!")
                
            cur_loc_name = f'x_conv{i}'

            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.occ_loss_loc:
                distill_return[cur_loc_name] = x
                distill_return[cur_loc_name + '_gt'] = self.create_occ_gt(x, gt_box_list)

            laterals.append(x)
        if self.multiscale_out: z1 = voxel_to_point(x, z0)
        laterals = laterals[:-1][::-1]

        decoder_outs = []

        for i, decoder_layer in enumerate(self.decoder):
            cur_loc_name = f'x_deconv{i}'
            x = decoder_layer[0](x)

            if check_nan(self.sparseconv_backend, x):
                print("NaN detected in x!")
                
            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.occ_loss_loc:
                distill_return[cur_loc_name] = x
                distill_return[cur_loc_name + '_gt'] = self.create_occ_gt(x, gt_box_list)

            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, laterals[i]))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(
                    x, torch.cat((x.features, laterals[i].features), dim=1))
            elif self.sparseconv_backend == 'minkowski':
                x = ME.cat(x, laterals[i])

            x = decoder_layer[1](x)        
            decoder_outs.append(x)
        
        if self.multiscale_out:
            if len(decoder_outs) >= 2:
                z2 = voxel_to_point(decoder_outs[1], z1)
            else:
                z2 = voxel_to_point(decoder_outs[-1], z1)
            z3 = voxel_to_point(decoder_outs[-1], z2)
            # LiDAR 三尺度点级特征（深→中→浅）
            point_feat_lidar = torch.cat([z1.F, z2.F, z3.F], dim=1)
            decoder_outs[-1].F = point_feat_lidar
            distill_return['point_feat_lidar'] = point_feat_lidar

            # FoV 3D 分支（仅 TorchSparse）：点级对齐（__init__ 已初始化 UNet3D）
            if self.sparseconv_backend == 'torchsparse':
                x_fov_ms = locals().get('x_fov_ms', None)
                if x_fov_ms is not None and x_fov_ms.F.size(0) > 0:
                    # 一致性断言：FoV 输入通道应为 519
                    assert x_fov_ms.F.shape[1] == self.fov_in_dim, \
                        f"FoV in_dim mismatch: expected {self.fov_in_dim}, got {x_fov_ms.F.shape[1]}"
                    # 将子模块放到与主张量同设备
                    self.lidar_backbone_fov = self.lidar_backbone_fov.to(x.F.device)
                    out_fov, x4_fov, x2_fov, x0_fov = self.lidar_backbone_fov({'lidar_fov_ms': x_fov_ms})
                    # 与 TASeg 对齐：深→中→浅 对应 z0→z1→z2 的点级查询
                    z1_fov = voxel_to_point(x4_fov, z0)
                    z2_fov = voxel_to_point(x2_fov, z1)
                    z3_fov = voxel_to_point(x0_fov, z2)
                    point_feat_fov = torch.cat([z1_fov.F, z2_fov.F, z3_fov.F], dim=1)
                    distill_return['point_feat_fov'] = point_feat_fov
                    # 融合输入（LiDAR+FoV），按 TASeg 的六尺度拼接返回供上层头部使用
                    if 'point_feat_lidar' in distill_return:
                        assert distill_return['point_feat_lidar'].size(0) == point_feat_fov.size(0), \
                            'LiDAR 与 FoV 点数不一致，检查投影/索引对齐'
                        distill_return['fusion_feat'] = torch.cat([
                            distill_return['point_feat_lidar'], point_feat_fov
                        ], dim=1)
  
        distill_return['x_deconv4'] = x

        if 'x_deconv4' in self.occ_loss_loc and self.training:
            distill_return['x_deconv4'] = self.bev_out_3d(x)
            distill_return[cur_loc_name + '_gt'] = self.create_occ_gt(x, gt_box_list) 

        # return decoder_outs[-1].features, distill_return
    
        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features, distill_return['fusion_feat']
        elif self.sparseconv_backend == 'torchsparse':
            return decoder_outs[-1].F, distill_return['fusion_feat']
        else:
            raise NotImplementedError
