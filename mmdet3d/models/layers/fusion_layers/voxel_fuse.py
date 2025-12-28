
from typing import List, Union, Optional
import torch
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import mmcv
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmengine.model import BaseModule, constant_init, xavier_init
import torch.nn.functional as F
from .point_emb import PointsEmbedding

def sample_img_featues(
                    img_meta: dict,
                    img_features,
                    points_2d,
                    aligned: bool = True,
                    padding_mode: str = 'zeros',
                    align_corners: bool = True,
                    valid_flag: bool = False,
                    normalized: bool = True): 
    coor_x, coor_y = points_2d[:, 0][:, None], points_2d[:, 1][:, None]
    if len(points_2d.shape) == 3:
        depths = points_2d[:, 2]


    norm_coor_y = coor_y * 2 - 1
    norm_coor_x = coor_x * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
    
    vis = False
    if vis:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        # 可视化某个特征通道
        img_feat_0 = img_features[0, 0].detach().cpu().numpy()  # [H, W]
        plt.imshow(img_feat_0, cmap='viridis')
        H, W = img_feat_0.shape[0], img_feat_0.shape[1]
        x = ((grid[..., 0] + 1) / 2 * W).squeeze().cpu().numpy()  # [N]
        y = ((grid[..., 1] + 1) / 2 * H).squeeze().cpu().numpy()  # [N]

        # 可视化点（同上）
        plt.scatter(x, y, s=2, c='red')  # 采样点位置
        # plt.scatter(coor_x.cpu().numpy() * W, coor_y.cpu().numpy() * H, s=2, c='red')
        plt.title("Sampling Locations on Feature Map")
        plt.savefig('z.png')

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    if valid_flag and (len(points_2d.shape) == 3):
        # (N, )
        # valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
        #     coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
        #         depths > 0)
        valid = (norm_coor_x.squeeze() < 1) & (norm_coor_x.squeeze() > -1) & (
            norm_coor_y.squeeze() < 1) & (norm_coor_y.squeeze() > -1) & (
                depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)
    else:
        return point_features.squeeze().t().cuda()

 
@MODELS.register_module()
class pass2d_fusion(BaseModule):
    def __init__(self, 
                 fusetype = 'pass_2d',
                 emb_cfg = None,
                 use_pts_leaner = True,
                 add_loc_emb = False,
                 input_lidar_dim = 256,
                 input_cam_dim = [256, 256, 256, 256, 256],
                 hiden_size = 256,
                 ):
        super(pass2d_fusion, self).__init__()
        self.fusetype = fusetype
        self.emb_cfg = emb_cfg

        if add_loc_emb and emb_cfg is not None:
            self.point_emb_layer = PointsEmbedding(emb_cfg)
        self.use_pts_leaner = use_pts_leaner
        self.add_loc_emb = add_loc_emb
        self.hiden_size = hiden_size

        if self.fusetype == 'dir_con':
            self.concat_compre_layer = nn.Sequential(nn.Linear(sum(input_cam_dim) + input_lidar_dim , input_lidar_dim))
        if self.fusetype == 'attention_map':
            self.fcs1_attention_map =  nn.Sequential(nn.Linear(sum(input_cam_dim) + input_lidar_dim , self.hiden_size))
            
        self.leaners = nn.Sequential(nn.Linear(input_lidar_dim, self.hiden_size))
        if add_loc_emb:
            if emb_cfg is not None:
                input_lidar_dim_loc = input_lidar_dim + emb_cfg.output_dim
            else:
                input_lidar_dim_loc = input_lidar_dim
            self.fcs1 = nn.Sequential(nn.Linear(sum(input_cam_dim) + self.hiden_size +  input_lidar_dim_loc, self.hiden_size))        
        else:
            self.fcs1 = nn.Sequential(nn.Linear(sum(input_cam_dim) + self.hiden_size , 
                                            self.hiden_size))
        self.fcs2 = nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size))
        self.fcs3 = nn.Sequential(nn.Linear(self.hiden_size, input_lidar_dim))

    def simple_sample_single_2d(self, 
                                img_feats, 
                                reference_points_cams, 
                                batch_input_metas=None):
        sam_loc = reference_points_cams
        img_choice = sample_img_featues(batch_input_metas, img_feats, sam_loc)
        return img_choice # num_p , num_channel

    def obtain_mlvl_feats(self, 
                          img_dict, 
                          reference_points_cams, # bs, num_p, num_imglevel, 2
                          batch_input_metas):

        img_feats_per_point = []

        bs_id = 0 # 前面取出batch
        assert reference_points_cams.shape[0] == 1
        mlvl_img_feats = []

        img_feats = img_dict['img_feats_cur_batch']

        for level in range(reference_points_cams.shape[-2]):
            mlvl_img_feats.append(
                self.simple_sample_single_2d(
                    img_feats = img_feats[level], 
                    reference_points_cams = reference_points_cams[bs_id][:, level, :3], 
                    batch_input_metas = batch_input_metas[bs_id]))
        mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
        img_feats_per_point = mlvl_img_feats
        return img_feats_per_point
        
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                img_data_dict = None,
                batch_input_metas = None,
                xyz = None,
                **kwargs) -> torch.Tensor:
        # correspondence
        pts_feat = query.squeeze(0)
        img_feats_per_point = self.obtain_mlvl_feats(
            img_data_dict, 
            reference_points,
            batch_input_metas) # n_p, level_channnel_add

        # modality fusion
        if self.fusetype == 'pass_2d':
            if self.use_pts_leaner:
                feat_learner = F.relu(self.leaners(pts_feat))
            else:
                feat_learner = self.leaners(pts_feat)
            if self.add_loc_emb:
                if self.emb_cfg is not None:
                    loc_info = self.point_emb_layer(xyz)
                    loc_info = torch.cat([loc_info, pts_feat], 1)
                else:
                    loc_info = pts_feat
                feat_cat = torch.cat([img_feats_per_point, feat_learner, loc_info], 1)
            else:
                feat_cat = torch.cat([img_feats_per_point, feat_learner], 1)
            feat_cat = self.fcs1(feat_cat)
            attention_map = torch.sigmoid(self.fcs2(feat_cat))
            fuse_feat = F.relu(feat_cat * attention_map)
            fuse_feat = self.fcs3(fuse_feat)

        elif self.fusetype == 'dir_con':
            feat_cat = torch.cat([img_feats_per_point, pts_feat], 1)
            fuse_feat = self.concat_compre_layer(feat_cat)
        
        elif self.fusetype == 'attention_map':
            feat_cat = torch.cat([img_feats_per_point, pts_feat], 1)
            feat_cat = self.fcs1_attention_map(feat_cat)
            attention_map = torch.sigmoid(self.fcs2(feat_cat))
            fuse_feat = F.relu(feat_cat * attention_map)
            fuse_feat = self.fcs3(fuse_feat)           

        return fuse_feat.unsqueeze(0), None ,None

@MODELS.register_module()
class pmf_fusion(pass2d_fusion):
    def __init__(
            self,
            use_pts_leaner = False,
            pass2d_output = False,
            input_cam_dim = [256,256,256,256,256],
            input_lidar_dim = 256,
            hiden_size = 256
                ):
        super(pmf_fusion, self).__init__()

        self.use_pts_leaner = use_pts_leaner
        self.pass2d_output = pass2d_output
        self.hiden_size = input_lidar_dim

        if self.use_pts_leaner:
            self.hiden_size = hiden_size
            self.leaners = nn.Sequential(nn.Linear(input_lidar_dim, self.hiden_size))


        pcd_channels = input_lidar_dim
        img_channels = sum(input_cam_dim)
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(self.hiden_size + img_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(pcd_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv1d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(pcd_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(pcd_channels),
            nn.Sigmoid()
        )

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                img_data_dict = None,
                batch_input_metas = None,
                **kwargs) -> torch.Tensor:
        pts_feat = query.squeeze(0)
        img_feats_per_point = self.obtain_mlvl_feats(
            img_data_dict, 
            reference_points,
            batch_input_metas)

        pts_feat = pts_feat

        if self.use_pts_leaner:
            pts_feat = F.relu(self.leaners(pts_feat))

        feat_cat = torch.cat((pts_feat, img_feats_per_point), dim=1)
        feat_cat = feat_cat.transpose(0, 1).unsqueeze(0)  # [C, N] -> [1, C, N]
        fuse_out = self.fuse_conv(feat_cat)
        attention_map = self.attention(fuse_out)
        
        attention_map = attention_map.squeeze(0).transpose(0, 1)  # -> [N, C]
        fuse_out = fuse_out.squeeze(0).transpose(0, 1)

        if self.pass2d_output:
            fuse_feat = F.relu(fuse_out * attention_map)
        else:
            fuse_feat = fuse_out * attention_map + pts_feat

        return fuse_feat.unsqueeze(0), None ,None
    
    