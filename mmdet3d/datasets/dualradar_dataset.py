# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .seg3d_dataset import Seg3DDataset
from mmengine.fileio import load
from os import path as osp
from mmdet3d.structures import get_box_type

@DATASETS.register_module()
class DualRadarDataset(Seg3DDataset):
    # vod
    METAINFO = {
        'classes': ('background', 'Car', 'Tricycle', 'Cyclist', 'Motorcycle', 'Dense_bicycle', 'Pedestrian', 'Bus', 'Construction', 'Truck', 'Traffic_signs', 'Huge_vehicle'),
        # color for vis
        'palette': [
            (0, 0, 0),  # Index 0

            (0, 255, 0),  # Index 1: Car green 
            (0, 0, 255),  # Index 2: Tricycle blue
            (255, 0, 0),  # Index 3: Cyclist red  

            (150, 60, 30),  # Index 4: Motorcycle Brownish 棕色
            (255, 255, 255),  # Index 5: Dense_bicycle 白色
            (180, 30, 80),  # Index 6: Pedestrian Dark Pink

            (255, 255, 0),  # Index 7: Bus 纯黄
            (255, 255, 255),  # Index 8: Construction
            (122, 139, 139),  # Index 9: Truck 浅绿色 

            (90, 30, 150),  # Index 10: Traffic_signs Purple 深紫色
            (66, 66, 66),  # Index 11: Huge_vehicle 灰色

            # (255, 255, 255),  # Index 12: vehicle_others no_point_in_val
            # (255, 255, 255),  # Index 13: ride_uncertain no_point_in_val
        ],
        'seg_valid_class_ids':
        tuple(range(12)),
        'seg_all_class_ids':
        tuple(range(12)),
        'ignore_index': 12,
    }


    def __init__(self,
                 data_root: str,
                 
                 ann_file: str,
                 
                 pts_semantic_label_path = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 # dual radar
                 pcd_limit_range: List[float] = [-40, -10, -3, 40, 60.4,  1],
                 
                 metainfo = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 ignore_index  = None,
                 scene_idxs = None,
                 **kwargs) -> None:
        self.pts_semantic_label_path = pts_semantic_label_path
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')
        self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])
        

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        raw_data_list = annotations

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list
    
    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            if 'plane' in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        def info_pcdet2mmdet(info):
            mmdet_dict = dict()
            mmdet_dict['lidar_points'] = dict()
            mmdet_dict['lidar_points']['lidar_path'] = info['point_cloud']['lidar_idx'] + '.bin'
            mmdet_dict['lidar_points']['num_pts_feats'] = info['point_cloud']['num_features']
            mmdet_dict['pts_semantic_mask_path'] = self.pts_semantic_label_path
            return mmdet_dict

        info = info_pcdet2mmdet(info)

        info = super().parse_data_info(info, default_cam_key='CAM2')

        # info['ann_info'] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict: # nuscenes cbgs
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if 'label' in ann_name and ann_name != 'attr_label':
                    temp_anns = [
                        self.seg_label_mapping[item] for item in temp_anns
                    ]
                    
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns

                if mapped_ann_name == 'gt_bboxes_3d':
                    lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
                    # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
                    gt_bboxes_3d = CameraInstance3DBoxes(
                        ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                            np.linalg.inv(lidar2cam))
                    ann_info[mapped_ann_name] = gt_bboxes_3d

            ann_info['instances'] = info['instances']

        return ann_info

    def get_cat_ids(self, idx: int):
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``CBGSDataset`` or ``ClassBalancedDataset``requires a subclass
        which implements this method.

        Args:
            idx (int): The index of data.

        Returns:
            set[int]: All categories in the sample of specified index.
        """
        info = self.get_data_info(idx)
        info['ann_info'] = self.parse_ann_info(info)
        gt_labels = info['ann_info']['gt_labels_3d'].tolist()
        return set(gt_labels)