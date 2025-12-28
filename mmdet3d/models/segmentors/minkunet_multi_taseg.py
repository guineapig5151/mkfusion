# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
import os
import pickle
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmengine.logging import print_log
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
import torch.nn.functional as F
import torch.nn as nn
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor
from .encoder_decoder import EncoderDecoder3D
from .mono.tools.test_scale_cano import get_depth_feat_in_model

@MODELS.register_module()
class MinkUNet_multi_TASeg(EncoderDecoder3D):
    def __init__(self,
                 freeze_img = True,
                 use_depth_map = False,
                 input_depth_channel = 386,
                 out_depth_channel = 8,
                 depth_map_cfg = None,
                 input_cam_dim = None,
                 pts_fusion_layer: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs) # self.backbone 
        self.freeze_img = freeze_img
        self.use_depth_map = use_depth_map
        self.input_cam_dim = input_cam_dim
        self.depth_map_cfg = depth_map_cfg

        if pts_fusion_layer:
            self.pts_fusion_layer = MODELS.build(pts_fusion_layer)
            
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
            self.with_img_backbone = True
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
            self.with_img_neck = True

        if img_rpn_head is not None:
            self.img_rpn_head = MODELS.build(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = MODELS.build(img_roi_head)
        
        if self.use_depth_map:
            self.depth_convs = nn.ModuleList()
            for i in range(len(self.input_cam_dim)):  # 假设你知道img_feats有几层
                depth_conv = nn.Sequential(
                    nn.Conv2d(input_depth_channel, out_depth_channel, kernel_size=3, stride=1, padding=1),  # 1通道depth变成8通道
                    nn.BatchNorm2d(8),
                    nn.ReLU(inplace=True)
                )
                self.depth_convs.append(depth_conv)

        if self.use_depth_map and self.depth_map_cfg is not None:
            from .mono.model.monodepth_model import get_configured_monodepth_model
            from .mono.utils.running import load_ckpt

            config_path = "mmdet3d/models/segmentors/metric3d_config.yaml"
            from mmengine import Config

            # 从 YAML 文件读取配置
            cfg = Config.fromfile(config_path)
            self.cfg = cfg
            depth_model = get_configured_monodepth_model(cfg, )
            depth_model = torch.nn.DataParallel(depth_model).cuda()
            depth_model, _,  _, _ = load_ckpt(cfg.load_from, depth_model, strict_match=False)
            self.depth_model = depth_model

    def init_weights(self):
        """Initialize model weights."""
        super(MinkUNet_multi_TASeg, self).init_weights()

        if self.freeze_img:
            if self.with_img_backbone:
                print_log('freezed image backbone', 'current')
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                print_log('freezed image neck', 'current')
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def process_and_concat_depth(self, img_feats, depth_map):
        depth_maps_resized = []
        img_feats_with_depth = []

        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)

        for idx, cur_level_img_feat in enumerate(img_feats):
            _, _, h, w = cur_level_img_feat.shape
            depth_map_resize = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)
            depth_map_processed = self.depth_convs[idx](depth_map_resize)

            img_feat_with_depth = torch.cat([cur_level_img_feat, depth_map_processed], dim=1)
            img_feats_with_depth.append(img_feat_with_depth)
            depth_maps_resized.append(depth_map_resize)

        return img_feats_with_depth

    def extract_img_feat(self, 
                         img: Tensor, 
                         depth_map, # 2, 1216, 1936
                         input_metas: List[dict]) -> dict:
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

        if self.use_depth_map:
            if self.depth_map_cfg is not None:
                if self.depth_map_cfg['type'] == 'metricv2':
                    data_list = get_depth_feat_in_model(img, input_metas, self.depth_model, self.cfg)
                    depth_map = torch.stack([d['pred_depth'] for d in data_list], dim=0)
                    # bs, 1216,1936
                    ret_feat = torch.cat([d['output']['ref_feat'] for d in data_list], dim=0) # torch.Size([2, 386, 154, 266])
                    depth_map = ret_feat

            depth_maps_resized = []
            img_feats_with_depth = []
            for cur_level_img_feat in img_feats:
                _, _, h, w = cur_level_img_feat.shape  # 当前特征图的空间尺寸 (B, C, H, W)

                if depth_map.dim() == 3:  # [B, H, W]
                    depth_map = depth_map.unsqueeze(1)  # [B, 1, H, W]
                    
                # 将depth_map resize到(h, w)
                # depth_map_resize = F.interpolate(
                #     depth_map, size=(h, w), mode='bilinear', align_corners=False
                # )
                # depth_maps_resized.append(depth_map_resize)
                img_feats_with_depth = self.process_and_concat_depth(img_feats, depth_map)
            return img_feats_with_depth
        
        return img_feats
    
    def extract_voxel_feat(self, batch_inputs_dict: dict, img_feats, batch_input_metas, imgs, **kwargs) -> Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        x, student_return = self.backbone(voxel_dict['voxels'], 
                          voxel_dict['coors'],
                          img_feats,
                          batch_input_metas,
                          imgs)
        if self.with_neck:
            x = self.neck(x)
        return x, student_return

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas=None) -> tuple:
        imgs = batch_inputs_dict.get('imgs', None)

        bs = imgs.shape[0]
        depth_map = []

        if self.use_depth_map:
            for batch_id in range(bs):
                depth_map_cur_batch = batch_input_metas[batch_id]['depth_map']
                depth_map_cur_batch = torch.tensor(depth_map_cur_batch).cuda().unsqueeze(0)
                depth_map.append(depth_map_cur_batch)
            depth_map = torch.cat(depth_map)
            # imgs = torch.stack(imgs).float() # yyt_c
        else:
            depth_map = None

        img_feats = self.extract_img_feat(imgs, 
                                          depth_map,
                                          batch_input_metas)

        pts_feats, pts_feats_fusion = self.extract_voxel_feat(
            batch_inputs_dict,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            imgs=imgs)
        return pts_feats, pts_feats_fusion

    def loss(self, inputs: dict, data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_metas = [item.metainfo for item in data_samples]
        x, x_fused = self.extract_feat(inputs, batch_input_metas)
        losses = self.decode_head.loss(dict(x=x,x_fused=x_fused), data_samples, self.train_cfg)
        return losses
    
    def predict(self, inputs: dict, batch_data_samples):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x, x_fused = self.extract_feat(inputs, batch_input_metas)
        seg_logits_list = self.decode_head.predict(dict(x=x,x_fused=x_fused), batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples, points=inputs['points'])