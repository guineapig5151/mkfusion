# Copyright (c) OpenMMLab. All rights reserved.
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


@MODELS.register_module()
class MinkUNet_distill(MinkUNet):
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

        sparse_choice_type_list = sparse_choice_type.split('_')
        if sparse_choice_type_list[0] == 'agg':
            self.sparse_choice = sparse_agg_fuse(sparse_choice_type_list[1], sparse_choice_type_list[2])   
        else:
            self.sparse_choice = sparse_cat_fuse()       

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
        voxels_main = voxel_dict['voxels']
        voxel_input = torch.cat((voxels_main, voxel_sweep), dim=0)
        coor_input = torch.cat((voxel_dict['coors'], coors_sweep), dim=0)
        num_main = voxels_main.size(0)        # 主体素数量  
        num_sweep = voxel_sweep.size(0)      # sweep体素数量  
        mask = torch.zeros(num_main + num_sweep, dtype=torch.bool)  
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

        teacher_radar_tensor = self.sparse_choice.forward(choice_teacher_feat, choice_student_feat)

        loss_cur_layer = 0

        if 'kl' in self.distill_type[layer_name]:
            from .distill_utils import kl_distill
            teacher_radar_tensor = self.sparse_choice.forward(teacher_return[layer_name], student_return[layer_name])
            teacher_logit = self.teacher_decode_head.forward(teacher_radar_tensor.features)
            student_logit = self.decode_head.forward(student_return[layer_name].features)
            loss = kl_distill(teacher_logit, student_logit) * self.distill_weight.get('kl',1)
            loss_cur_layer += loss

        if 'mse' in self.distill_type[layer_name]:
            loss = F.mse_loss(teacher_radar_tensor.features, choice_student_feat.features) * self.distill_weight.get('mse',1)
            loss_cur_layer += loss

        if 'l1' in self.distill_type[layer_name]:
            loss = F.l1_loss(teacher_radar_tensor.features, choice_student_feat.features) * self.distill_weight.get('l1',1)
            loss_cur_layer += loss

        if 'aff' in self.distill_type[layer_name]:
            loss_model = Affinity_Loss()
            loss = loss_model.forward(choice_student_feat, teacher_radar_tensor) * self.distill_weight.get('aff',1)
            loss_cur_layer += loss

        if 'qfl' in self.distill_type[layer_name]:
            loss_model = Quality_Focal_Loss_no_reduction()
            loss = loss_model.forward(choice_student_feat.features, teacher_radar_tensor.features) * self.distill_weight.get('qfl',1)
            loss_cur_layer += loss
            
        if 'rd' in self.distill_type[layer_name]:
            loss_model = RDLoss()
            loss = loss_model.forward(choice_student_feat, teacher_radar_tensor) * self.distill_weight.get('rd', 1)
            loss_cur_layer += loss
        return loss_cur_layer
    
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