# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval, seg_eval_12

from mmdet3d.registry import METRICS
import torch

@METRICS.register_module()
class SegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 data_type = '2',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.data_type = data_type
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(SegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for batch_id, data_sample in enumerate(data_samples):
            if self.data_type == '12':
                points = {'points': data_sample['points']}
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v

            if self.data_type == '12':
                self.results.append((eval_ann_info, cpu_pred_3d, points))
            elif self.data_type == '123':
                self.results.append((eval_ann_info, cpu_pred_3d, data_sample['pred_pts_seg_top2']['pred_pts_seg_top2'].cpu().numpy(), data_sample['prob_pts_seg_top2']['prob_pts_seg_top2'].cpu().numpy()))
            else:
                self.results.append((eval_ann_info, cpu_pred_3d))
            
            vis_2d = False
            if vis_2d:
                import os 
                import numpy as np
                import matplotlib.pyplot as plt
                # 手动定义 RGB 颜色 (红, 橙, 黄, 绿, 蓝, 靛, 紫)
                rainbow_colors = [
                    (1, 0, 0),    # 红
                    (1, 0.5, 0),  # 橙
                    (1, 1, 0),    # 黄
                    (0, 1, 0),    # 绿
                    (0, 0, 1),    # 蓝
                    (0.29, 0, 0.51),  # 靛
                    (0.5, 0, 0.5)  # 紫
                ]
                # 转换为 NumPy 数组
                color_map = np.array(rainbow_colors)

                voxel_point = data_batch['inputs']['points'][batch_id]
                seg_pred = cpu_pred_3d['pts_semantic_mask'].astype(np.int64)
                seg_gt = eval_ann_info['pts_semantic_mask'].astype(np.int64)
                wrong_mask = (seg_pred != seg_gt)
                points_3d = voxel_point[:, :3]
                proj_mat = data_sample['lidar2img']
                points_4 = torch.cat([points_3d, points_3d.new_ones(points_3d.shape[0], 1)], dim=-1)
                point_2d = points_4 @ proj_mat.T
                point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

                sam_loc = point_2d_res[wrong_mask].cpu().numpy()
                img = data_batch['inputs']['img'][batch_id]
                # img = img * std + mean
                img = img.permute(1,2,0).numpy().clip(0, 255).astype('uint8')
                plt.figure()
                img_rgb = img[..., ::-1]
                plt.imshow(img_rgb)

                # 获取 img 的宽高
                img_h, img_w = img.shape[:2]

                # 使用 np.clip 限制坐标范围
                sam_loc[:, 0] = np.clip(sam_loc[:, 0], 0, img_w - 1)  # 限制 x 坐标
                sam_loc[:, 1] = np.clip(sam_loc[:, 1], 0, img_h - 1)  # 限制 y 坐标

                plt.scatter(sam_loc[:, 0], sam_loc[:, 1], color = 'red', s=1.5)

                sam_loc = point_2d_res[~wrong_mask].cpu().numpy()
                sam_loc[:, 0] = np.clip(sam_loc[:, 0], 0, img_w - 1)  # 限制 x 坐标
                sam_loc[:, 1] = np.clip(sam_loc[:, 1], 0, img_h - 1)  # 限制 y 坐标
                plt.scatter(sam_loc[:, 0], sam_loc[:, 1], color = 'blue', s=1.5)

                name = data_sample['lidar_path'].split('/')[-1][:-4]
                save_dir = '_z_vis/'
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir)
                plt.savefig(save_dir + name + '.png')
                plt.clf()

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmcv.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']
        # need to map network output to original label idx
        cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
            np.int64)
        for original_label, output_idx in self.dataset_meta['label2cat'].items(
        ):
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            sample_idx = eval_ann['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int64)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f'{submission_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None


        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']

        gt_semantic_masks = []
        pred_semantic_masks = []
        mask_lidar = []

        if self.data_type != '12':
            if self.data_type == '123': # tj4d score alignment
                for eval_ann, sinlge_pred_results, pred_top2, prob_top2 in results:
                    gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
                    pred_top1 = sinlge_pred_results['pts_semantic_mask']
                    for cls_id in label2cat.keys():
                        if (cls_id == ignore_index) or (cls_id == 0):
                            continue
                        pred_top1[((pred_top1==0)&(pred_top2[1]==cls_id)&(prob_top2[1]>0.2))] = cls_id
                    pred_semantic_masks.append(pred_top1)
            else:
                for eval_ann, sinlge_pred_results in results:
                    gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
                    pred_semantic_masks.append(
                        sinlge_pred_results['pts_semantic_mask'])
                
        if self.data_type == '12':
            for eval_ann, sinlge_pred_results, points in results:
                gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
                pred_semantic_masks.append(
                    sinlge_pred_results['pts_semantic_mask'])
                points = points['points']
                cur_batch_mask_lidar = (points[:, 4] == 1) & (points[:, 5] == 1) & (points[:, 6] == 1)
                mask_lidar.append(cur_batch_mask_lidar)  
            return seg_eval_12(
                gt_semantic_masks,
                pred_semantic_masks,
                label2cat,
                ignore_index,
                logger=logger,
                mask_lidar= mask_lidar)
        
        ret_dict = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict
