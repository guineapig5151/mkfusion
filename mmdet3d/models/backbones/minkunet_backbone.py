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

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    from torchsparse import PointTensor
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    # x: SparseTensor, z: PointTensor
    # return: PointTensor
    def voxel_to_point(x, z, nearest=False):
        if z.idx_query is None or z.weights is None or z.idx_query.get(
                x.s) is None or z.weights.get(x.s) is None:
            off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
            old_hash = F.sphash(
                torch.cat([
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1)
                ], 1), off)
            pc_hash = F.sphash(x.C.to(z.F.device))
            idx_query = F.sphashquery(old_hash, pc_hash)
            weights = F.calc_ti_weights(z.C, idx_query,
                                        scale=x.s[0]).transpose(0, 1).contiguous()
            idx_query = idx_query.transpose(0, 1).contiguous()
            if nearest:
                weights[:, 1:] = 0.
                idx_query[:, 1:] = -1
            new_feat = F.spdevoxelize(x.F, idx_query, weights)
            new_tensor = PointTensor(new_feat,
                                    z.C,
                                    idx_query=z.idx_query,
                                    weights=z.weights)
            new_tensor.additional_features = z.additional_features
            new_tensor.idx_query[x.s] = idx_query
            new_tensor.weights[x.s] = weights
            z.idx_query[x.s] = idx_query
            z.weights[x.s] = weights

        else:
            new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
            new_tensor = PointTensor(new_feat,
                                    z.C,
                                    idx_query=z.idx_query,
                                    weights=z.weights)
            new_tensor.additional_features = z.additional_features

        return new_tensor

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME
import copy
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
import numpy as np

def check_nan(backend, x):
    return (backend == 'spconv' and torch.isnan(x.features).any()) or (backend == 'torchsparse' and torch.isnan(x.F).any())

@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
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
                 multiscale_out: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.multiscale_out = multiscale_out
        if multiscale_out:
            assert sparseconv_backend == 'torchsparse', 'Currently, only torchsparse backend support multiscale out'
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

    def forward(self, 
                voxel_features: Tensor, 
                coors: Tensor, 
                in_distill=False,
                gt_box_list = None,
                ):
        distill_return = {}

        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
            z = PointTensor(x.F, x.C.float())
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
            decoder_outs[-1].F = torch.cat([z1.F, z2.F, z3.F], dim=1)
  
        distill_return['x_deconv4'] = x

        if 'x_deconv4' in self.occ_loss_loc and self.training:
            distill_return['x_deconv4'] = self.bev_out_3d(x)
            distill_return[cur_loc_name + '_gt'] = self.create_occ_gt(x, gt_box_list) 

        # return decoder_outs[-1].features, distill_return
    
        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features, distill_return
        elif self.sparseconv_backend == 'torchsparse':
            return decoder_outs[-1].F, distill_return
        else:
            raise NotImplementedError