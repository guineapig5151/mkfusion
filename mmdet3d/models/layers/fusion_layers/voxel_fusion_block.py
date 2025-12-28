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

@MODELS.register_module()
class VoxelFusionBlock(BaseModule):
    def __init__(self,
                 img_cross_att: Optional[dict] = 
                    dict(type='RadarImageCrossAttention',
                        query_embed_dims=256,
                        value_embed_dims=256,
                        output_embed_dims=256,
                        deformable_attention=dict(
                            type='MSDeformableAttention',
                            num_levels=4
                            ),
                    ),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.img_cross_att = MODELS.build(img_cross_att)

    def forward(self, x, 
                batch_mask_list, 
                centroids_img_list, 
                img_data_dict,
                batch_input_metas,
                point_xyz_list = None):
        feat_flatten = img_data_dict['feat_flatten']
        spatial_shapes = img_data_dict['spatial_shapes']
        level_start_index = img_data_dict['level_start_index']

        # extract feature from image
        bs = feat_flatten.shape[0]
        updated_query = torch.zeros(x.features.shape, dtype=x.features.dtype, device=x.features.device)
        sample_locs_all = torch.zeros([x.features.shape[0],4*8*len(level_start_index),2], dtype=x.features.dtype, device=x.features.device)
        sample_weights_all = torch.zeros([x.features.shape[0],4*8*len(level_start_index),1], dtype=x.features.dtype, device=x.features.device)

        for batch_id_cur in range(bs):
            batch_mask = batch_mask_list[batch_id_cur]
            non_empty_voxel_feats = x.features[batch_mask]
            
            ref_pts = centroids_img_list[batch_id_cur][None, :, None, :].repeat(1, 1, len(spatial_shapes), 1)

            img_data_dict['img_feats_cur_batch'] = [
                img_data_dict_cur[batch_id_cur:batch_id_cur+1, ...]
                for img_data_dict_cur in img_data_dict['img_feats']
            ]
            assert img_data_dict['img_feats_cur_batch'][0].shape[0]

            ms_feats, sample_locs, sample_weights = self.img_cross_att(
                            query=non_empty_voxel_feats.unsqueeze(0),
                            key=feat_flatten[batch_id_cur].unsqueeze(0),
                            value=feat_flatten[batch_id_cur].unsqueeze(0),
                            reference_points_cams=ref_pts,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            img_data_dict = img_data_dict,
                            batch_input_metas = batch_input_metas,
                            xyz = point_xyz_list[batch_id_cur][:,1:],
                        )
            updated_query[batch_mask] = ms_feats[0]
            if sample_locs is not None:
                sample_locs_all[batch_mask] = sample_locs[0]
                sample_weights_all[batch_mask] = sample_weights[0]
        return updated_query, sample_locs_all, sample_weights_all


@MODELS.register_module()
class RadarImageCrossAttention(BaseModule):
    def __init__(self,
                 query_embed_dims=256,
                 value_embed_dims=256,
                 output_embed_dims=256,
                 num_cams=1,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=True,
                 deformable_attention=dict(
                     type='MSDeformableAttention',
                     query_embed_dims=256,
                     value_embed_dims=256,
                     output_embed_dims=256,
                     num_levels=4),
                 ):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        #self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        if deformable_attention['type'] == 'MSDeformableAttention':
            deformable_attention.update(query_embed_dims=query_embed_dims,
                                        value_embed_dims=value_embed_dims,
                                        output_embed_dims=output_embed_dims,)
        self.deformable_attention = MODELS.build(deformable_attention)
        self.query_embed_dims = query_embed_dims
        self.value_embed_dims = value_embed_dims
        self.output_embed_dims = output_embed_dims
        self.num_cams = num_cams
        #self.output_proj = nn.Linear(value_embed_dims, output_embed_dims)
        self.batch_first = batch_first
        #self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        pass
        #xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                spatial_shapes=None,
                reference_points_cams=None,
                level_start_index=None,
                img_data_dict = None,
                batch_input_metas = None,
                xyz = None):
        """Forward Function of Detr3DCrossAtten.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_key, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_key, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            tpv_masks (List[Tensor]): The mask of each views.
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            reference_points_cams (List[Tensor]): The reference points in
                each camera.
            tpv_masks (List[Tensor]): The mask of each views.
            level_start_index (List[int]): The start index of each level.

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        updated_query, sample_locs, sample_weights = self.deformable_attention(
            query=query,
            key=key,
            value=value,
            reference_points=reference_points_cams, # torch.Size([1, 517, 5, 2])
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_data_dict = img_data_dict,
            batch_input_metas = batch_input_metas,
            xyz = xyz,
        )

        return updated_query, sample_locs, sample_weights



@MODELS.register_module()
class MSDeformableAttention(BaseModule):
    def __init__(self,
                 query_embed_dims: int = 256,
                 value_embed_dims: int = 256,
                 output_embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 weight_act_func: Optional[str] = 'softmax',
                 residual: Optional[bool] = True,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if value_embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {value_embed_dims} and {num_heads}')
        dim_per_head = value_embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.weight_act_func = weight_act_func
        self.residual = residual
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.query_embed_dims = query_embed_dims
        self.value_embed_dims = value_embed_dims
        self.output_embed_dims = output_embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            query_embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(query_embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(value_embed_dims, value_embed_dims)
        self.output_proj = nn.Linear(value_embed_dims, output_embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        #constant_init(self.attention_weights, val=0, bias=1/(self.num_levels * self.num_points))
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

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
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        if self.weight_act_func == 'softmax':
            attention_weights = attention_weights.softmax(-1)
        elif self.weight_act_func == 'sigmoid':
            attention_weights = attention_weights.sigmoid()
        elif self.weight_act_func == 'relu':
            attention_weights = attention_weights.relu()
        else:
            raise NotImplementedError
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return (self.dropout(output) + identity) if self.residual else self.dropout(output), sampling_locations.view(bs, num_query, -1, 2), attention_weights.view(bs, num_query, -1, 1)

