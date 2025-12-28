# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D
import torch
from ..get_before_tem import get_before_tem_feat

@MODELS.register_module()
class MinkUNet_tem(EncoderDecoder3D):
    r"""MinkUNet is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self, tem_num, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get_before_tem = get_before_tem_feat(tem_num=tem_num)

    def loss(self, inputs: dict, data_samples: SampleList):
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
        x, _, data_samples = self.extract_feat(inputs, data_samples)
        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        return losses

    def predict(self, inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        x, _, batch_data_samples = self.extract_feat(inputs, batch_data_samples)
        seg_logits_list = self.decode_head.predict(x, batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples, inputs['points'])

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. Defaults to None.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(x)

    def extract_before_time_feat(self, batch_inputs_dict):
        batch_inputs_dict = self.data_preprocessor(batch_inputs_dict, self.training)
        voxel_dict = batch_inputs_dict['inputs']['voxels']
        x, distill_return = self.backbone(voxel_dict['voxels'], voxel_dict['coors'], in_distill=True) 
        return x, distill_return

    def extract_feat(self, batch_inputs_dict: dict, data_samples=None) -> Tensor:

        sweep_point_list = [data_sample.sweep_point_list.tensor for data_sample in data_samples]
        batch_inputs_dict_tem = self.get_before_tem(batch_inputs_dict, sweep_point_list, self)
        
        # data proce
        data_cur = {}
        data_cur['inputs'] = {}
        data_cur['inputs']['points'] = batch_inputs_dict['points']
        data_cur['data_samples'] = data_samples
        batch_inputs_dict = data_cur
        batch_inputs_dict = self.data_preprocessor(batch_inputs_dict, self.training)
        data_samples = batch_inputs_dict['data_samples']

        voxel_dict = batch_inputs_dict['inputs']['voxels']
        x, distill_return = self.backbone(voxel_dict['voxels'], voxel_dict['coors'], before_feat_dict=batch_inputs_dict_tem, in_distill=True)
      
        if self.with_neck:
            x = self.neck(x)
            
        return x, distill_return, data_samples

    def train_step(self, data,
                   optim_wrapper):
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            # data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def test_step(self, data) -> list:
        # data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore
    
    def val_step(self, data) -> list:
        # data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore