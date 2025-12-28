import torch
import torch.nn as nn
from .dca import MSDeformableAttention
from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential, SparseConv2d
from mmdet3d.models.layers.sparse_block import (replace_feature)
from mmdet3d.registry import MODELS

@MODELS.register_module()
class Dca_input_stacom(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = model_cfg.get('input_channels', 512)
        self.dca_layer_num = model_cfg.get('num_dca_layer', 1)
        self.attn_list = nn.ModuleList()
        for _ in range(self.dca_layer_num):
            cur_attn_layer = MSDeformableAttention(
                     query_embed_dims=input_channels,
                     value_embed_dims=input_channels,
                     output_embed_dims=input_channels,
                     num_levels=4                
            )
            self.attn_list.append(cur_attn_layer)
        if model_cfg.get('FPN_CFG', None) is not None:
            self.fpn = BaseBEVBackbone(**model_cfg.FPN_CFG)
        else:
            self.fpn = None
        
        self.temporal_conv = SparseSequential(
            SparseConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=False, indice_key='temporal_conv'),
            nn.BatchNorm1d(input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
    
    def get_dense_fuse_sp_feat(self, sparse_features, voxel_indices, spatial_shape, dense_tensor, j = 0):
        batch_size = dense_tensor.shape[0]
        W, H = spatial_shape # xyz
        output_feat_all_batch = torch.zeros_like(sparse_features)
        for bs in range(batch_size):
            batch_mask = voxel_indices[:, 0] == bs
            indices = voxel_indices[batch_mask][:, 1:3] # xyz
            query = sparse_features[batch_mask]
            value = dense_tensor[bs]
            ref_pts = indices.float()
            ref_pts[:, 0] /= W
            ref_pts[:, 1] /= H # xyz
            # (C, H*W)
            cur_sp_shape = torch.tensor(dense_tensor.shape[-2:], device=dense_tensor.device)
            cur_sp_shape = cur_sp_shape.unsqueeze(0)
            value = value.view((value.shape[0], -1)).permute(1, 0)
            output,_,_ = self.attn_list[j](
                query=query.unsqueeze(0),
                key=value.unsqueeze(0),
                value=value.unsqueeze(0),
                # 注意在sparse tensor.indice yx(hw)，但是attn xy # the newest xy
                reference_points=ref_pts[:,[0, 1]].unsqueeze(-2).unsqueeze(0), # -2 level 0 bs
                spatial_shapes=cur_sp_shape,
                level_start_index=torch.tensor([0], device=dense_tensor.device),
            )
            output_feat_all_batch[batch_mask] = output[0]
        return output_feat_all_batch
         
    def forward(self, x, before_feat_dict, cur_loc_name): # xyz
        prev_feats = []
        before_feat_list = before_feat_dict['before_feat_list']
        for i in range(len(before_feat_list)):
            prev_feats.append(before_feat_list[i][cur_loc_name])
        sparse_tensor = x

        agg_feat_list = [sparse_tensor.features]

        for prev_feat in prev_feats:
            dense_val_feat = prev_feat.dense().squeeze(-1)
            if self.fpn is not None:
                real_dense_val_feat = self.fpn({'spatial_features': dense_val_feat})['spatial_features_2d']
            else:
                real_dense_val_feat = dense_val_feat
            query_feat = sparse_tensor.features
            for j in range(self.dca_layer_num):
                query_feat = self.get_dense_fuse_sp_feat(query_feat, sparse_tensor.indices[:, :-1], sparse_tensor.spatial_shape[:-1], real_dense_val_feat, j)
            agg_feat_list.append(query_feat)

        # tem_feat = torch.cat(agg_feat_list, dim=-1)
        tem_feat = agg_feat_list[0] + agg_feat_list[1]
        return tem_feat
