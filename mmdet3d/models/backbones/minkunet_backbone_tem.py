# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List
import os
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
import numpy as np

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

from .minkunet_backbone import MinkUNetBackbone
from .voxel_fusion_utils import sparse_cat_fuse

@MODELS.register_module()
class MinkUNetBackbone_tem(MinkUNetBackbone):
    def __init__(self,
                 tem_fuse_block,
                 tem_fuse_loc = ['x_conv5'],
                 tem_fuse_type = 'st',
                 spatial_shape = [1024, 1024, 41], # xyz
                 voxel_size = None,
                 point_cloud_range = None,
                 query_voxel_fuse_method = 'add',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.spatial_shape = torch.tensor(spatial_shape).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.tem_fuse_loc = tem_fuse_loc
        self.tem_fuse_type = tem_fuse_type
        self.tem_fuse_blocks = nn.ModuleDict()
        self.sparse_cat_fuse = sparse_cat_fuse()

        for i, loc in enumerate(self.tem_fuse_loc):
            if self.tem_fuse_type == 'st':
                self.tem_fuse_blocks[loc] = MODELS.build(tem_fuse_block)
            elif self.tem_fuse_type == 'dca':
                self.tem_fuse_blocks[loc] = MODELS.build(tem_fuse_block)
 
    def forward(self, voxel_features, coors, before_feat_dict=None, in_distill=False):
        if in_distill == True:
            distill_return = {}

        assert self.sparseconv_backend == 'spconv'
        spatial_shape_ = coors.max(0)[0][1:] + 1
        spatial_shape = self.spatial_shape
        zyx_input = False
        # 检查 spatial_shape 是否每个元素都大于等于 spatial_shape_
        assert (spatial_shape >= spatial_shape_).all(), \
            f"Error: spatial_shape {spatial_shape} must be greater than or equal to spatial_shape_ {spatial_shape_}"
        

        batch_size = int(coors[-1, 0]) + 1
        x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                batch_size)
        
        tensor = x.indices[:, 0]
        assert torch.all(tensor[:-1] <= tensor[1:])

        x = self.conv_input(x)

        laterals = [x]
        for i, cur_encoder_layer in enumerate(self.encoder):
            cur_loc_name = f'x_conv{i+1}' # fuse_after
            x = cur_encoder_layer(x)

            # distill before fuse
            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.tem_fuse_loc and before_feat_dict is not None:
                tem_fuse_tensor = self.tem_fuse_blocks[cur_loc_name](x, before_feat_dict, cur_loc_name)
                if isinstance(tem_fuse_tensor, Tensor):
                    x = x.replace_feature(tem_fuse_tensor)
                else:
                    x_tem = self.sparse_cat_fuse(tem_fuse_tensor, x)
                    x = x.replace_feature(x_tem.features)
                
            assert torch.isnan(x.features).any() == False
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            cur_loc_name = f'x_deconv{i+1}' # fuse_after
            assert torch.isnan(x.features).any() == False

            x = decoder_layer[0](x)  # x_  = self.decoder[0][0](x)

            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.tem_fuse_loc and before_feat_dict is not None:
                tem_fuse_tensor = self.tem_fuse_blocks[cur_loc_name](x, before_feat_dict, cur_loc_name)
                if isinstance(tem_fuse_tensor, Tensor):
                    x = x.replace_feature(tem_fuse_tensor)
                else:
                    x_tem = self.sparse_cat_fuse(tem_fuse_tensor, x)
                    x = x.replace_feature(x_tem.features)
                  
            x = replace_feature(x, torch.cat((x.features, laterals[i].features), dim=1))
            x = decoder_layer[1](x)
            decoder_outs.append(x)
            
        if in_distill == True:
            distill_return['x_deconv4'] = x    
            return decoder_outs[-1].features, distill_return
        
        return decoder_outs[-1].features

