# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

from mmcv.cnn.bricks import ConvModule
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import PointFPModule
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from einops import rearrange
from .decode_head import Base3DDecodeHead
from ..layers.pointnet_modules.stanet_sa_module import mlps

class FeaturePropogate(nn.Module):
    def __init__(self, interpolated_num, layer_config, weight_eps: float = 1e-8) -> None:
        super().__init__()
        # Bind interpolation neighbors to constructor arg (passed from decode_head.num_neighbor)
        # Keep default behavior identical (current configs pass 15)
        self.interpolated_num = int(interpolated_num)
        self.interpolated_mode = 'l2'
        self.dist_mode = 'l2'
        self.interpolated_fq=  2
        self.neighbor_n = int(interpolated_num)
        # Epsilon for inverse-distance weighting; default keeps legacy behavior
        self.weight_eps = float(weight_eps)
        # Optional radius threshold (in meters) to gate neighbors; None keeps original behavior
        self.radius_threshold = None

        self.mlp_1_0 = None
        self.mlp_2_0 = None
        self.mlp_2_1 = None

        self.mlp_2_1 = mlps(320, [640, 640])
        self.mlp_1_0 = mlps(704, [640, 768])

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape) # B, N, n
        view_shape[1:] = [1] * (len(view_shape) - 1) # B,1,1
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1 # 1, N, n
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # B,N,n
        new_points = points[batch_indices, idx, :]
        return new_points


    def forward(self,res):
        dist = self.interpolated_mode # 'max'
        high_feature, inter_feature, low_feature = res['radar_feat'], res['l1_cls_feat'], res['l2_cls_feat']
        interpolated_layers = self.interpolated_fq
        if interpolated_layers == 1:

            sorted_weights,sorted_idx = torch.sort(res['l0_to_l2'],-1) # B,N,M
            sorted_weights,sorted_idx = sorted_weights[:,:,:self.interpolated_num], sorted_idx[:,:,:self.interpolated_num] # B,N,K
            
            if dist == 'l2':
                # optional radius gating (default None -> no-op)
                if self.radius_threshold is not None:
                    mask = (sorted_weights <= self.radius_threshold)
                    # ensure at least one neighbor kept
                    fallback = (~mask).all(dim=2, keepdim=True)
                    mask = (mask | fallback).float()
                else:
                    mask = 1.0
                dist_recip = (1.0 / (sorted_weights + self.weight_eps)) * mask
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / (norm + self.weight_eps)

                weight = rearrange(weight,'b n k -> b n k 1')
                intermediate_feature = self.index_points(low_feature, sorted_idx) * weight
                intermediate_feature = torch.sum(intermediate_feature,dim=2)
            
            intermediate_feature = torch.cat([inter_feature,intermediate_feature],dim=2)
            intermediate_feature = rearrange(intermediate_feature,'b n c-> b c n')
            intermediate_feature = self.mlp_2_0(intermediate_feature)
            intermediate_feature = rearrange(intermediate_feature,'b c n-> b n c')


        # __________________________________________________________________________
        if interpolated_layers == 2:
            dist_map_0_1, dist_map_1_2 = res['l0_to_l1'], res['l1_to_l2']
            dist_map_0_1 = dist_map_0_1.permute(0,2,1)
            dist_map_1_2 = dist_map_1_2.permute(0,2,1)

            sorted_weights,sorted_idx = torch.sort(dist_map_1_2,-1)
            sorted_weights,sorted_idx = sorted_weights[:,:,:self.neighbor_n], sorted_idx[:,:,:self.neighbor_n]
            if dist == 'l2':
                if self.radius_threshold is not None:
                    mask = (sorted_weights <= self.radius_threshold)
                    fallback = (~mask).all(dim=2, keepdim=True)
                    mask = (mask | fallback).float()
                else:
                    mask = 1.0
                dist_recip = (1.0 / (sorted_weights + self.weight_eps)) * mask
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / (norm + self.weight_eps)

                weight = rearrange(weight,'b n k -> b n k 1')
                intermediate_feature = self.index_points(low_feature, sorted_idx) * weight
                intermediate_feature = torch.sum(intermediate_feature,dim=2)

            elif dist == 'max':
                sorted_idx = sorted_idx[...,0]
                intermediate_feature = self.index_points(low_feature,sorted_idx)

            elif dist == 'tf_feature':
                pass
                
            elif dist == 'mean':
                
                sorted_weights,sorted_idx = sorted_weights[:,:,:3], sorted_idx[:,:,:3]
                
                intermediate_feature = self.index_points(low_feature, sorted_idx)
                #intermediate_feature = intermediate_feature[:,:,0,:]*self.weight_0 + intermediate_feature[:,:,0,:]*self.weight_0
                intermediate_feature = torch.mean(intermediate_feature,dim=2)
            
            intermediate_feature = torch.cat([inter_feature,intermediate_feature],dim=2)
            intermediate_feature = rearrange(intermediate_feature,'b n c-> b c n')
            intermediate_feature = self.mlp_2_1(intermediate_feature)
            intermediate_feature = rearrange(intermediate_feature,'b c n-> b n c')

            # -----------------------------------------------------------------
            sorted_weights,sorted_idx = torch.sort(dist_map_0_1,-1)
            # use the same K for the second hop to keep behavior controlled by configuration
            sorted_weights,sorted_idx = sorted_weights[:,:,:self.neighbor_n], sorted_idx[:,:,:self.neighbor_n]

            if self.radius_threshold is not None:
                mask = (sorted_weights <= self.radius_threshold)
                fallback = (~mask).all(dim=2, keepdim=True)
                mask = (mask | fallback).float()
            else:
                mask = 1.0

            dist_recip = (1.0 / (sorted_weights + self.weight_eps)) * mask
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / (norm + self.weight_eps)

            weight = rearrange(weight,'b n k -> b n k 1')
            intermediate_feature = self.index_points(intermediate_feature, sorted_idx) * weight
            interpolated_feat = torch.sum(intermediate_feature,dim=2)

        if high_feature is not None:
                new_points = torch.cat([high_feature, interpolated_feat], dim=-1)

                new_points = rearrange(new_points, 'b n c -> b c n')
                new_points = self.mlp_1_0(new_points)
                #new_points = rearrange(new_points, 'b c n -> b n c')
                # new_points = interpolated_feat
        else: # 最后一层新的点为插值结果 B,N,C
            new_points = interpolated_feat
        
        res['x'] = new_points
        return res

    def _up_sample(self,feature, dist, mode):
            
            sorted_weights,sorted_idx = torch.sort(dist,-1)
            if mode == 'l2':
                sorted_weights,sorted_idx = sorted_weights[:,:,:5], sorted_idx[:,:,:5]
                if self.radius_threshold is not None:
                    mask = (sorted_weights <= self.radius_threshold)
                    fallback = (~mask).all(dim=2, keepdim=True)
                    mask = (mask | fallback).float()
                else:
                    mask = 1.0
                dist_recip = (1.0 / (sorted_weights + self.weight_eps)) * mask
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / (norm + self.weight_eps)

                weight = rearrange(weight,'b n k -> b n k 1')
                intermediate_feature = self.index_points(feature, sorted_idx) * weight
                intermediate_feature = torch.sum(intermediate_feature,dim=2)

            elif dist == 'max':
                sorted_idx = sorted_idx[...,0]
                intermediate_feature = self.index_points(feature,sorted_idx)

            #elif dist == 'tf_feature':
            #    assert len(low_points) == 3
            
            return intermediate_feature
    

@MODELS.register_module()
class STAHead(Base3DDecodeHead):
    r"""PointNet2 decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (Sequence[Sequence[int]]): Tuple of mlp channels in FP
            modules. Defaults to ((768, 256, 256), (384, 256, 256),
            (320, 256, 128), (128, 128, 128, 128)).
        fp_norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers used
            in FP modules. Defaults to dict(type='BN2d').
    """

    def __init__(self,
                 num_neighbor: int = 15,
                 radius_threshold: float = None,
                 weight_eps: float = 1e-8,
                 eps: float = None,
                 **kwargs) -> None:
        super(STAHead, self).__init__(**kwargs)
        # keep backward-compat alias: `eps` if provided overrides weight_eps
        if eps is not None:
            weight_eps = float(eps)

        self.FP_module = FeaturePropogate(num_neighbor, dict(), weight_eps=weight_eps)
        # Expose optional radius gating via config; default None keeps original behavior
        self.FP_module.radius_threshold = radius_threshold

        del self.conv_seg
        semantic_in_dim = 768
        self.conv_seg = nn.Sequential(
            nn.Linear(semantic_in_dim, semantic_in_dim//2),
            nn.LayerNorm([semantic_in_dim//2,]),
            nn.ReLU(),
            nn.Linear(semantic_in_dim//2, self.num_classes),
        )

    def _extract_input(self,
                       feat_dict: dict) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Coordinates and features of
            multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        output = self.FP_module(feat_dict)
        x = output['x']
        x = rearrange(x, 'b c n -> b n c')
        x = self.conv_seg(x)
        x = rearrange(x, 'b n c -> b c n')

        return x
