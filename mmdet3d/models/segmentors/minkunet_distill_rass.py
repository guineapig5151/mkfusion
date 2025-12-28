# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
from .minkunet import MinkUNet
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from spconv.pytorch import SparseConvTensor

from .distill_utils import sparse_cat_fuse, \
                            sparse_agg_fuse, \
                            Affinity_Loss, \
                            Quality_Focal_Loss_no_reduction, \
                            RDLoss
import torch.nn.functional as F

class SparseAlignIntersect(nn.Module):
    """
    Align two spconv SparseConvTensor by the intersection of their indices
    and return two tensors with identical indices (the intersection). Features
    are gathered from the respective inputs;不存在的行不会出现，因此无需填充。
    """

    def forward(self, t1, t2) -> Tuple[object, object]:
        # Sanity checks
        assert t1.indices.shape[1] == t2.indices.shape[1], 'index dim mismatch'
        assert tuple(t1.spatial_shape) == tuple(t2.spatial_shape), 'spatial_shape mismatch'
        assert int(t1.batch_size) == int(t2.batch_size), 'batch_size mismatch'
        # Keep dtype consistent for indices on both inputs (spconv usually int32)
        assert t1.indices.dtype == t2.indices.dtype, 'indices dtype mismatch'

        def _empty_out():
            D = t1.indices.shape[1]
            out_idx_common = torch.zeros((0, D), dtype=t1.indices.dtype, device=t1.indices.device)
            feat1 = t1.features.new_zeros((0, t1.features.shape[1]))
            feat2 = t2.features.new_zeros((0, t2.features.shape[1]))
            out1 = t1.__class__(feat1, out_idx_common, t1.spatial_shape, t1.batch_size)
            out2 = t2.__class__(feat2, out_idx_common.to(t2.indices.device), t2.spatial_shape, t2.batch_size)
            return out1, out2

        # Fast path: if either is empty, intersection is empty
        if t1.indices.numel() == 0 or t2.indices.numel() == 0:
            return _empty_out()

        # Row-wise intersection via sort + searchsorted (GPU-friendly, concise)
        # Supports indices of shape [N, D] where D >= 3 and the first column is batch id.
        # Common cases:
        #   - 4D indices: [b, z, y, x], spatial_shape = [Z, Y, X]
        #   - 3D indices: [b, x, y]   (你的需求)，spatial_shape = [X, Y]
        compute_device = t1.features.device
        D = int(t1.indices.shape[1])
        assert D >= 3, f'Expect indices with at least 3 dims ([b, ...]), got {D}'

        # Move indices to compute device and linearize to 1D keys (int64)
        idx1_dev = t1.indices.to(compute_device)
        idx2_dev = t2.indices.to(compute_device)
        shape = [int(v) for v in t1.spatial_shape]
        assert len(shape) == D - 1, (
            f'spatial_shape rank mismatch: indices has {D-1} spatial dims, '
            f'but spatial_shape has {len(shape)}')
        # Generic linearization: key = (((b * S0) + d0) * S1 + d1) * ...
        def lin_key(idx):
            key = idx[:, 0].to(torch.long)  # batch id
            for i, Si in enumerate(shape):
                di = idx[:, i + 1].to(torch.long)
                key = key * int(Si) + di
            return key

        k1 = lin_key(idx1_dev)
        k2 = lin_key(idx2_dev)

        # Only sort keys2; keep k1 in original order to preserve t1 order
        o2 = torch.argsort(k2)
        keys2 = k2[o2]
        pos2 = o2  # positions in t2

        # For each key in k1 (original order), find match in sorted keys2
        loc = torch.searchsorted(keys2, k1)
        locc = torch.clamp(loc, max=keys2.numel() - 1)
        valid = (loc < keys2.numel()) & (keys2[locc] == k1)
        if not torch.any(valid):
            return _empty_out()
        # i1: indices in t1 (preserve original order); i2: mapped indices in t2
        i1 = torch.nonzero(valid, as_tuple=False).view(-1)
        i2 = pos2[locc[valid]]

        # Build outputs
        out_idx_dev = idx1_dev.index_select(0, i1)
        feat1 = t1.features.index_select(0, i1)
        feat2 = t2.features.index_select(0, i2)

        # Construct outputs (positional ctor for spconv v2)
        # Keep indices identical in content; place them on each tensor's indices.device
        out1 = t1.__class__(feat1, out_idx_dev.to(t1.indices.device), t1.spatial_shape, t1.batch_size)
        out2 = t2.__class__(feat2, out_idx_dev.to(t2.indices.device), t2.spatial_shape, t2.batch_size)
        return out1, out2




@MODELS.register_module()
class MinkUNet_distill_RaSS(MinkUNet):
    def __init__(self, 
                 
                 teacher_backbone,
                 teacher_decode_head,
                 freeze_teacher = True,
                 sparse_choice_type = '',
                 teacher_cp_file = None,
                 distill_type = {},
                 distill_weight = 10,
                 seg_loss_weight = 1.0,
                
                 voxel_encoder = None,
                fuse_after = False,
                 freeze_img = True,
                 pts_fusion_layer = None,
                 img_backbone = None,
                 img_neck = None,  
                 with_dis_mlp = False,               
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.freeze_img = freeze_img
        self.fuse_after = fuse_after

        if voxel_encoder is not None:
            self.voxel_encoder = MODELS.build(voxel_encoder)
        else:
            self.voxel_encoder = None

        if pts_fusion_layer:
            self.pts_fusion_layer = MODELS.build(pts_fusion_layer) 
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
            self.with_img_backbone = True
        else:
            self.with_img_backbone = False

        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False

        self.teacher_backbone = MODELS.build(teacher_backbone)
        self.teacher_decode_head = MODELS.build(teacher_decode_head)
        
        self.freeze_teacher = freeze_teacher
        if self.freeze_teacher == True:
            self.teacher_backbone = self.load_and_freeze(self.teacher_backbone, teacher_cp_file, 'backbone')
            self.teacher_decode_head = self.load_and_freeze(self.teacher_decode_head, teacher_cp_file, 'decode_head')

        # distill loss
        self.distill_type = distill_type
        self.distill_loc = list(distill_type.keys())
        self.distill_weight = distill_weight
        self.seg_loss_weight = seg_loss_weight

        self.channel_map = dict(
            x_conv0=32, x_conv1=64, x_conv2=128, x_conv3=256, x_deconv0=256, x_deconv1=128, x_deconv2=96, x_deconv3=96, x_deconv4=96
        )
        self.dis_mapping = nn.ModuleDict()
        self.with_dis_mlp = with_dis_mlp
        if self.with_dis_mlp:
            for k in self.distill_loc:
                self.dis_mapping[k] = nn.Linear(self.channel_map[k], self.channel_map[k])
        else:
            for k in self.distill_loc:
                self.dis_mapping[k] = nn.Identity()

        sparse_choice_type_list = sparse_choice_type.split('_')
        self.sparse_choice = SparseAlignIntersect()
        # if sparse_choice_type_list[0] == 'agg':
        #     self.sparse_choice = sparse_agg_fuse(sparse_choice_type_list[1], sparse_choice_type_list[2])   
        # else:
        #     self.sparse_choice = sparse_cat_fuse()       

    def load_and_freeze(self, model, teacher_cp_file, prefix):
        # student img
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

        checkpoint = _load_checkpoint(
            teacher_cp_file,
            map_location='cpu')
        checkpoint_state_dict = checkpoint["state_dict"]
        metadata = getattr(checkpoint_state_dict, '_metadata', OrderedDict())

        prefix += '.'
        prefix_len = len(prefix)
        state_dict = OrderedDict({
            k[prefix_len:]: v
            for k, v in checkpoint_state_dict.items() if k.startswith(prefix)
        })
        metadata = OrderedDict({
            k[prefix_len:]: v
            for k, v in metadata.items() if k.startswith(prefix)
        })
        state_dict._metadata = metadata
        load_state_dict(model, state_dict, strict=True, logger=None)
        # .eval() ?
        self.freeze_model(model)
        model.eval()
        return model
    
    def freeze_model(self, model):
        if hasattr(model, "module"):
            for p in model.module.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = False


    def extract_img_feat(self, img: Tensor, input_metas) -> dict:
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_voxel_feat_multi(self, batch_inputs_dict: dict, img_feats, batch_input_metas, imgs, in_distill, **kwargs) -> Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        x, distill_return = self.backbone(voxel_dict['voxels'], 
                          voxel_dict['coors'],
                          img_feats,
                          batch_input_metas,
                          imgs,
                          in_distill = in_distill)
        if self.with_neck:
            x = self.neck(x)
        return x, distill_return

    def extract_feat_multi_student(self, batch_inputs_dict: dict,
                     batch_input_metas=None) -> tuple:
        imgs = batch_inputs_dict.get('imgs', None)
        # imgs = torch.stack(imgs).float() # yyt_c
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        pts_feats, distill_return = self.extract_voxel_feat_multi(
            batch_inputs_dict,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            imgs=imgs,
            in_distill = True)
        return pts_feats, distill_return
    
    def extract_feat_student(self, batch_inputs_dict: dict, be_student=True) -> Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        x, student_return = self.backbone(voxel_dict['voxels'], voxel_dict['coors'], be_student)
        return x, student_return

    def extract_feat_teacher(self, batch_inputs_dict: dict, be_teacher=True) -> Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        voxel_sweep = voxel_dict['voxelssweep_point_list']
        coors_sweep = voxel_dict['coorssweep_point_list']
        voxel_input = voxel_sweep
        coor_input = coors_sweep
        num_main = 0
        num_sweep = voxel_sweep.size(0)
        mask = torch.zeros(num_sweep, dtype=torch.bool)  
        mask[:num_main] = True
        mask[num_main:] = False 
        
        x, teacher_return = self.teacher_backbone(voxel_input, coor_input, be_teacher)

        # x = x[mask]
        # coor_test = teacher_return['x_deconv4'].indices[mask]
        # assert torch.all(coor_test - voxel_dict['coors']) == 0
    
        return x, teacher_return
        
    def sparse_distill(self, teacher_return, student_return, layer_name, data_samples=None):
        choice_teacher_feat = teacher_return[layer_name]
    
        if self.fuse_after == True:
            choice_student_feat = student_return[layer_name + 'fuse_after']
        else:
            choice_student_feat = student_return[layer_name]

        t1, t2 = self.sparse_choice.forward(self.bev_out(choice_teacher_feat), self.bev_out(choice_student_feat))
        # t1, t2 = self.sparse_choice.forward(choice_teacher_feat, choice_student_feat)

        loss_cur_layer = 0
        loss = F.mse_loss(t1.features, self.dis_mapping[layer_name](t2.features)) * self.distill_weight[layer_name]
        loss_cur_layer += loss

        return loss_cur_layer
    
    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

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
    
    def loss(self, inputs: dict, data_samples: SampleList):

        if self.voxel_encoder is not None:
            voxel_input = inputs['voxels']['voxels']
            coor_input = inputs['voxels']['coors']
            voxel_features, feature_coors = self.voxel_encoder(
                voxel_input, coor_input, data_samples)
            inputs['voxels']['voxels'], inputs['voxels']['coors'] = voxel_features, feature_coors

        # seg_label = self.decode_head._stack_batch_gt(data_samples)
        if 'imgs' in inputs:
            x, student_return = self.extract_feat_multi_student(inputs, batch_input_metas = [item.metainfo for item in data_samples])
        else:
            x, student_return = self.extract_feat_student(inputs)

        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        x_teacher, teacher_return = self.extract_feat_teacher(inputs)

        distill_loss = 0
        for layer_id in range(len(self.distill_loc)):
            distill_loss += self.sparse_distill(teacher_return, student_return, self.distill_loc[layer_id], data_samples)
        
        if len(self.distill_loc) > 0:
            losses['distill_loss'] = (distill_loss / len(self.distill_loc))
        losses['loss_sem_seg'] *= self.seg_loss_weight
        
        return losses

    def predict(self, inputs: dict,
                batch_data_samples: SampleList) -> SampleList:

        if 'imgs' in inputs:
            x, student_return = self.extract_feat_multi_student(inputs, batch_input_metas = [item.metainfo for item in batch_data_samples])
        else:
            x, student_return = self.extract_feat_student(inputs)
        seg_logits_list = self.decode_head.predict(x, batch_data_samples)
        
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples, inputs['points'])