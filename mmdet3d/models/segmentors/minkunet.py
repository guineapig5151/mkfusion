# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D
import torch
import torch.nn.functional as F
from torch import nn as nn

@MODELS.register_module()
class MinkUNet(EncoderDecoder3D):
    def __init__(self, 
                 use_msvfe = False,
                 voxel_encoder = None,
                 test_radar_input=False, 
                 use_sweep_point_list=False, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_msvfe = use_msvfe
        if voxel_encoder is not None:
            self.voxel_encoder = MODELS.build(voxel_encoder)
        else:
            self.voxel_encoder = None
        self.test_radar_input = test_radar_input
        self.use_sweep_point_list = use_sweep_point_list
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # 自动包含 Sigmoid
        self.occ_loss_loc_list = self.backbone.occ_loss_loc
        channel_map = {
            'x_conv1': 64,
            'x_deconv4': 96,
            'x_conv3': 256,
        }
        for i, occ_loss_loc in enumerate(self.occ_loss_loc_list):
            seg_channel = channel_map[occ_loss_loc]
            self.conv_seg2 = nn.Linear(seg_channel, 1)

    def loss(self, inputs: dict, data_samples: SampleList):
        gt_box_list = []

        for batch_id in range(len(data_samples)):
            gt_box = data_samples[batch_id].gt_instances_3d
            gt_box_list.append(gt_box)

        x, distill_return, point2voxel_map = self.extract_feat(inputs, gt_box_list)

        if len(self.occ_loss_loc_list):
            loss_occ = 0
            for i in range(len(distill_return) // 2):
                occ_pred = self.conv_seg2(list(distill_return.values())[i].features)
                occ_gt = list(distill_return.values())[i + 1]
                occ_gt = torch.cat(occ_gt)
                occ_gt = occ_gt.gt(0).float().unsqueeze(-1)  # 用 .gt() 直接转换为 {0,1}，然后转 float
                loss_occ += self.loss_fn(occ_pred, occ_gt)
        
        if self.use_msvfe:  
            for data_sample in data_samples:
                if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask \
                        = data_sample.gt_pts_seg.pts_semantic_mask                  
            x = x[point2voxel_map]

        losses = self.decode_head.loss(x, data_samples, self.train_cfg)

        if len(self.occ_loss_loc_list):
            losses['loss_occ'] = loss_occ
        return losses 

    def predict(self, inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        gt_box_list = []
        for batch_id in range(len(batch_data_samples)):
            gt_box = batch_data_samples[batch_id].gt_instances_3d
            gt_box_list.append(gt_box)

        x, _, point2voxel_map = self.extract_feat(inputs, gt_box_list)

        if self.use_msvfe:
            x = x[point2voxel_map]
            for batch_id, data_sample in enumerate(batch_data_samples):
                res_voxel_coors = inputs['points'][batch_id]
                batch_id_col = torch.full((res_voxel_coors.shape[0], 1), batch_id, dtype=res_voxel_coors.dtype, device=res_voxel_coors.device)
                data_sample.batch_idx = batch_id_col.squeeze(-1)
                data_sample.point2voxel_map = torch.arange(res_voxel_coors.shape[0])
                if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask \
                        = data_sample.gt_pts_seg.pts_semantic_mask   
                    
        seg_logits_list = self.decode_head.predict(x, batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples, inputs['points'])

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        x = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(x)

    def extract_feat(self, batch_inputs_dict: dict, gt_box_list=None, data_sample=None) -> Tensor:
        if self.use_msvfe == False:
            voxel_dict = batch_inputs_dict['voxels']
            voxel_input = voxel_dict['voxels']
            coor_input = voxel_dict['coors']

            # 检查 voxel_input 是否包含 NaN
            if torch.isnan(voxel_input).any():
                print("NaN detected in voxel_input!")

            # 检查 coor_input 是否包含 NaN
            if torch.isnan(coor_input).any():
                print("NaN detected in coor_input!")


        else:
            batch_points = []
            for batch_id, points in enumerate(batch_inputs_dict['points']):
                batch_id_col = torch.full((points.shape[0], 1), batch_id, dtype=points.dtype, device=points.device)
                batch_points.append(torch.cat([batch_id_col, points], dim=1))  # 在第一列添加 batch_id
                
            # 沿着数据维度（点数维度）拼接
            points = torch.cat(batch_points, dim=0)  # 最终形状 [N_total, C+1]
            voxel_input = points
            coor_input = points

        if self.voxel_encoder is not None:
            voxel_features, feature_coors, point2voxel_map = self.voxel_encoder(
                voxel_input, coor_input, data_sample)
            voxel_input, coor_input = voxel_features, feature_coors
        
        if self.use_sweep_point_list == True:
            if self.training == False and self.test_radar_input:
                pass
            else:
                voxel_sweep = voxel_dict['voxelssweep_point_list']
                coors_sweep = voxel_dict['coorssweep_point_list']
                voxels_main = voxel_dict['voxels']
                voxel_input = torch.cat((voxels_main, voxel_sweep), dim=0)
                coor_input = torch.cat((voxel_dict['coors'], coors_sweep), dim=0)
                # 构造掩码：主体素部分用 True，sweep 部分用 False  
                num_main = voxels_main.size(0)        # 主体素数量  
                num_sweep = voxel_sweep.size(0)      # sweep体素数量  
                mask = torch.zeros(num_main + num_sweep, dtype=torch.bool)  
                mask[:num_main] = True
                mask[num_main:] = False 

        x, distill_return = self.backbone(voxel_input, coor_input)

        if self.use_sweep_point_list:
            if self.training == False and self.test_radar_input:
                pass
            else:
                x = x[mask]
                coor_test = distill_return['x_deconv4'].indices[mask]
                assert torch.all(coor_test - voxel_dict['coors']) == 0

        if self.with_neck:
            x = self.neck(x)
        if self.voxel_encoder is not None:
            return x, distill_return, point2voxel_map

        return x, distill_return, None
