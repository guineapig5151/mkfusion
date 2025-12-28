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
from .voxel_fusion_utils import proj_to_img
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type

@MODELS.register_module()
class MinkUNetBackbone_multi(MinkUNetBackbone):

    def __init__(
                self,
                fusion_locations,
                seg_locations = ['x_conv1'],
                fusion_block_type = 'deform',
                pass2d_fusion_fusetype = 'pass_2d',
                input_cam_dim = [256, 256, 256, 256, 256],
                spatial_shape = [1024, 1024, 41], # xyz
                fuse_cfg = {},
                query_voxel_fuse_method = 'add',
                add_loc_emb = False,
                loc_emb_type = 'pts_feat',
                use_pts_leaner = True,
                pass2d_output = False,
                emb_cfg = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.seg_locations = seg_locations
        
        channel_map = {
            'x_conv0': 32,
            'x_conv1': 64,
            'x_conv3': 256,
            'x_deconv3':96,
            'x_deconv4': 96,
        }

        self.query_voxel_fuse_method = query_voxel_fuse_method
        self.spatial_shape = torch.tensor(spatial_shape).cuda()
        self.coord_type = 'LIDAR'
        self.fusion_locations = fusion_locations
        self.fuse_cfg = fuse_cfg
        self.fusion_block_type = fusion_block_type
        self.fusion_blocks = nn.ModuleDict()

        self.img_levels_to_fuse = self.fuse_cfg.get('img_levels_to_fuse', [0, 1, 2, 3, 4])
        self.img_dim = self.fuse_cfg.get('img_dim', 256)
        self.deform_weight_act_fun = self.fuse_cfg.get('deform_weight_act_fun', 'softmax')
        self.deform_block_residual = self.fuse_cfg.get('deform_block_residual', True)
        
        for i, loc in enumerate(self.fusion_locations):
            # fuse method

            if self.fusion_block_type == 'dff':
                fusion_block = dict(type='VoxelFusionBlock',
                                    img_cross_att = \
                                dict(type='RadarImageCrossAttention',
                                    query_embed_dims=channel_map[loc],#256
                                    value_embed_dims=self.img_dim,
                                    output_embed_dims=channel_map[loc],#256
                                    deformable_attention=dict(
                                        type='MSDeformableAttention',
                                        num_levels=len(self.img_levels_to_fuse),
                                        weight_act_func=self.deform_weight_act_fun,
                                        residual=self.deform_block_residual
                                        ),
                                    )
                                    )
            elif self.fusion_block_type[:6] == 'pass2d':
                fusion_block = dict(type='VoxelFusionBlock',
                                    img_cross_att = \
                                dict(type='RadarImageCrossAttention',
                                    query_embed_dims=channel_map[loc],#256
                                    value_embed_dims=self.img_dim,
                                    output_embed_dims=channel_map[loc],#256
                                    deformable_attention=dict(
                                        type='pass2d_fusion',
                                        fusetype = pass2d_fusion_fusetype,
                                        emb_cfg = emb_cfg,
                                        use_pts_leaner = use_pts_leaner,
                                        add_loc_emb = add_loc_emb,
                                        input_cam_dim = input_cam_dim,
                                        input_lidar_dim = channel_map[loc],
                                        ),
                                    )
                                )       
            elif self.fusion_block_type == 'pmf':
                fusion_block = dict(type='VoxelFusionBlock',
                                    img_cross_att = \
                                dict(type='RadarImageCrossAttention',
                                    query_embed_dims=channel_map[loc],#256
                                    value_embed_dims=self.img_dim,
                                    output_embed_dims=channel_map[loc],#256
                                    deformable_attention=dict(
                                        type='pmf_fusion',
                                        input_lidar_dim = channel_map[loc],
                                        use_pts_leaner = use_pts_leaner,
                                        pass2d_output = pass2d_output,
                                        ),
                                    )
                                )
            else:
                fusion_block = None
            
            self.fusion_blocks[loc] = MODELS.build(fusion_block)     
            

            ### query_voxel_fuse_method
            if self.query_voxel_fuse_method == 'dynamic_add':
                self.query_voxel_fuse_weight = nn.ParameterDict([(loc, nn.parameter.Parameter(torch.tensor(0.0), requires_grad=True)) for loc in self.fusion_locations])
            elif self.query_voxel_fuse_method == 'channel_dynamic_add':
                self.query_voxel_fuse_weight = nn.ParameterDict([(loc, nn.parameter.Parameter(torch.zeros([1,channel_map[loc]]), requires_grad=True)) for loc in self.fusion_locations])
            elif self.query_voxel_fuse_method == 'cat_and_map':
                self.query_voxel_fuse_map = nn.ModuleDict([(loc, nn.Sequential(
                    nn.Linear(channel_map[loc]*2, channel_map[loc]),
                    nn.BatchNorm1d(channel_map[loc]),
                    #nn.ReLU(inplace=True)
                    )) for loc in self.fusion_locations])
            elif self.query_voxel_fuse_method == 'loc_emb':
                coord_dim = 3
                self.query_loc_emb_layer = nn.ModuleDict([
                    (loc, nn.Sequential(
                        nn.Linear(channel_map[loc] + coord_dim, channel_map[loc]),  # 加上坐标维度
                        nn.BatchNorm1d(channel_map[loc]),
                        nn.ReLU(inplace=True)
                    )) for loc in self.fusion_locations
                ])

            
    def flatten_img_feats(self, img_feats):
        bs = img_feats[0].shape[0]
        dtype = img_feats[0].dtype
        device = img_feats[0].device
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            # (B, C, H, W) -> (B, C, HW) -> (B, HW, C)
            feat = feat.flatten(2).permute(0, 2, 1)
            feat = feat# + self.level_embeds[None, lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 1)  # bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        image_data_dict = {
            'img_feats': img_feats,
            'feat_flatten': feat_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index
        }
        return image_data_dict

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
        
    def get_multimodal_feat(self, 
            img_feats,
            sparse_query,
            batch_input_metas = None,
            loc = None,
            downsample_times = 2,
            imgs = None,
            zyx_input = False,
            **kwargs):
        img_data_dict = self.flatten_img_feats(img_feats)
        
        non_empty_coor = sparse_query.indices
        bs = imgs.shape[0]
        batch_mask_list = []
        for i in range(bs):
            batch_mask = (non_empty_coor[:, 0] == i)
            batch_mask_list.append(batch_mask)

        centroids_img_list, point_xyz_list = self.indice_proj2img(non_empty_coor, batch_input_metas, downsample_times, zyx_input)

        vis = False
        if vis:
            for i in range(bs):       
                # 获取 batch 中的 mask  
                sample_locs_ = centroids_img_list[i].detach().cpu().numpy()
                # sample_locs_ = np.clip(sample_locs_, [0, 0], [1, 1])
                # 获取第 i 张图片
                img = imgs[i]
                img = img * batch_input_metas[i]['img_std'] + batch_input_metas[i]['img_mean']

                # 将图像数据裁剪到 [0, 255] 的范围并转换为 uint8
                img = img.permute(1, 2, 0).cpu().numpy()
                if img.max() > 1:  # 如果是 [0, 255] 范围
                    img = img.clip(0, 255).astype('uint8')
                else:  # 如果是 [0, 1] 范围
                    img = (img * 255).clip(0, 255).astype('uint8')

                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(img)  # 显示裁剪后的图像

                # print("X coordinates:", sample_locs_[:, 0] * batch_input_metas[i]['batch_input_shape'][1])
                # print("Y coordinates:", sample_locs_[:, 1] * batch_input_metas[i]['batch_input_shape'][0])

                plt.scatter(
                    sample_locs_[:, 0] * batch_input_metas[i]['batch_input_shape'][1], 
                    sample_locs_[:, 1] * batch_input_metas[i]['batch_input_shape'][0], 
                    color='red', s=0.5  # 固定颜色
                )

                # 保存图片
                frame_id = batch_input_metas[i]['img_path'].split('/')[-1].split('.')[0]
                os.makedirs('./z_vis', exist_ok=True)
                plt.savefig(f'./z_vis/{frame_id}.png')
                plt.clf()

        new_features, sample_locs_all, sample_weights_all = self.fusion_blocks[loc](sparse_query, batch_mask_list, centroids_img_list, img_data_dict, batch_input_metas, point_xyz_list)
        return new_features

    def query_voxel_fuse(self, x, updated_query, method=None, loc=None):
        if method == 'add':
            new_features = torch.clone(x.features)
            new_features = updated_query + new_features

        elif method == 'nothing':
            # new_features = torch.clone(x.features)
            new_features = updated_query

        elif method == 'dynamic_add' or method == 'channel_dynamic_add':
            new_features = torch.clone(x.features)
            weight = self.query_voxel_fuse_weight[loc].sigmoid()
            new_features = weight * updated_query + (1 - weight) * new_features

        elif method == 'pure_cat' or method == 'cat_and_map':
            new_features = x.features.new_zeros([x.features.shape[0], x.features.shape[1]+updated_query.shape[1]])
            new_features[:, :x.features.shape[1]] = torch.clone(x.features)
            new_features[:, x.features.shape[1]:] = updated_query
            if method == 'cat_and_map':
                new_features = self.query_voxel_fuse_map[loc](new_features)

        elif method == 'loc_emb':
            # 假设 loc_feat 是 [N, C_feat]
            loc_feat = x.features  # e.g., [N, 64]
            loc_coords = x.indices[:,1:]  # e.g., [N, 3]
            # 拼接 [feature, coords] => [N, C_feat + 3]
            query_cat_loc = torch.cat([updated_query, loc_coords], dim=-1)
            # MLP: Linear(C_feat + 3, C_feat) + BN
            query_loc = self.query_loc_emb_layer[loc](query_cat_loc)  # -> [N, C_feat]
            # 更新 features
            new_features = updated_query + query_loc

        else:
            raise NotImplementedError(f'We do not implement {method} in query_voxel_fuse')
        return new_features
    
    def forward(self, 
                voxel_features, 
                coors, 
                img_feats, 
                batch_input_metas, 
                imgs, 
                in_distill=True,
                be_student = False,):
        if in_distill == True:
            distill_return = {}

        assert self.sparseconv_backend == 'spconv'
        spatial_shape_ = coors.max(0)[0][1:] + 1
        spatial_shape = self.spatial_shape
        # 检查 spatial_shape 是否每个元素都大于等于 spatial_shape_
        # assert (spatial_shape >= spatial_shape_).all(), \
        #     f"Error: spatial_shape {spatial_shape} must be greater than or equal to spatial_shape_ {spatial_shape_}"


        batch_size = int(coors[-1, 0]) + 1
        x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                batch_size)

        tensor = x.indices[:, 0]
        assert torch.all(tensor[1:] >= tensor[:-1])

        ori_spatial_shape = x.spatial_shape
        x = self.conv_input(x)
        cur_loc_name = 'x_conv_input'

        if in_distill == True:
            distill_return[cur_loc_name] = x

        laterals = [x]

        for i, cur_encoder_layer in enumerate(self.encoder):
            cur_loc_name = f'x_conv{i}' # fuse_after
            x = cur_encoder_layer(x)

            # distill before fuse
            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.fusion_locations and (be_student == False):
                cur_spatial_shape = x.spatial_shape
                downsample_times = ori_spatial_shape[0] // cur_spatial_shape[0]
                assert downsample_times % 2 == 0
                new_features = self.get_multimodal_feat(img_feats, x, batch_input_metas, cur_loc_name, downsample_times, imgs)
                assert torch.isnan(new_features).any() == False
                
                x = x.replace_feature(self.query_voxel_fuse(x, new_features, self.query_voxel_fuse_method, cur_loc_name))
            
            assert torch.isnan(x.features).any() == False
            laterals.append(x)
            
        laterals = laterals[:-1][::-1]

        decoder_outs = []

        for i, decoder_layer in enumerate(self.decoder):
            cur_loc_name = f'x_deconv{i}' # fuse_after
            assert torch.isnan(x.features).any() == False
            x = decoder_layer[0](x)      

            if in_distill == True:
                distill_return[cur_loc_name] = x

            if cur_loc_name in self.fusion_locations and (be_student == False):
                cur_spatial_shape = x.spatial_shape
                downsample_times = ori_spatial_shape[0] // cur_spatial_shape[0]
                assert downsample_times % 2 == 0 or downsample_times == 1
                new_features = self.get_multimodal_feat(img_feats, x, batch_input_metas, cur_loc_name, downsample_times, imgs)
                assert torch.isnan(new_features).any() == False
                
                x = x.replace_feature(self.query_voxel_fuse(x, new_features, self.query_voxel_fuse_method, cur_loc_name))

            if in_distill == True:
                distill_return[cur_loc_name + 'fuse_after'] = x

            x = replace_feature(x, torch.cat((x.features, laterals[i].features), dim=1))
            x = decoder_layer[1](x)
            decoder_outs.append(x)
            
        if in_distill == True:
            distill_return['x_deconv4'] = x    
            return decoder_outs[-1].features, distill_return
        
        return decoder_outs[-1].features

