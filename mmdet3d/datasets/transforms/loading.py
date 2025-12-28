# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union
import os
import mmcv
import mmengine
import numpy as np
from PIL import Image
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type
import torch
import cv2
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
@TRANSFORMS.register_module()
class Loaddepthmap(BaseTransform):
    def __init__(self,
                 depth_map_path,
                 **kwargs):
        self.depth_source = kwargs.get('depth_source', 'metric3dv2')
        self.depth_map_path = depth_map_path

    def get_depth_map(self, depth_map_path):
        if self.depth_source == 'metric3dv2':
            # 读取PNG
            depth_map_path = depth_map_path.replace('jpg', 'png')
            depth_png = Image.open(depth_map_path)  # depth_path是你的png路径
            depth_png = np.array(depth_png).astype(np.float32)

            # 恢复出原始深度（单位：米）
            depth_in_meters = depth_png / 256.0
            return depth_in_meters, None
        elif self.depth_source == 'moge':
            depth_path = depth_map_path.replace('jpg', 'exr')
            mask_path = depth_map_path.replace('.jpg', '_mask.png')
            depth = np.array(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)).astype(np.float32)
            depth[depth==np.inf] = -1
            valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return depth, np.array(valid_mask).astype(bool)
        elif self.depth_source == 'moge_pts':
            point_path = depth_map_path.replace('.jpg', '_pts.exr')
            mask_path = depth_map_path.replace('.jpg', '_mask.png')
            point_map = np.array(cv2.imread(str(point_path), cv2.IMREAD_UNCHANGED)).astype(np.float32)
            point_map[(point_map==np.inf).any(-1)] = -1
            valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return point_map.transpose(2, 0, 1), np.array(valid_mask).astype(bool)
        elif self.depth_source == 'dav2':
            depth_path = depth_map_path.replace('.jpg', '.png')
            depth = np.array(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)).astype(np.float32) / 255.0
            return depth[:, :, 0], None
        elif self.depth_source == 'dav2_metric':
            depth_path = depth_map_path.replace('.jpg', '.npy')
            depth = np.load(depth_path).astype(np.float32)
            return depth, None
        else:
            raise NotImplementedError
    
    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_path']
        depth_map_path = os.path.join(self.depth_map_path, filename.split('/')[-1])
        depth_map, depth_valid_mask = self.get_depth_map(depth_map_path)

        # test_depth_map_path = test_depth_root + filename.split('/')[-1]
        # test_depth_map = self.get_depth_map(test_depth_map_path)
        results['depth_map'] = depth_map
        results['depth_valid_mask'] = depth_valid_mask

        return results

@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(BaseTransform):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 5,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam = [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def transform(self, results: dict) -> dict:
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # TODO: load different camera image from data info,
        # for kitti dataset, we load 'CAM2' image.
        # for nuscenes dataset, we load 'CAM_FRONT' image.

        if 'CAM2' in results['images']:
            filename = results['images']['CAM2']['img_path']
            results['cam2img'] = results['images']['CAM2']['cam2img']
        elif len(list(results['images'].keys())) == 1:
            camera_type = list(results['images'].keys())[0]
            filename = results['images'][camera_type]['img_path']
            results['cam2img'] = results['images'][camera_type]['cam2img']
        else:
            raise NotImplementedError(
                'Currently we only support load image from kitti and '
                'nuscenes datasets')

        try:
            img_bytes = get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.
    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.
    Required Keys:
    - img
    Modified Keys:
    - img
    - img_path
    - img_shape
    - ori_shape
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweeps(BaseTransform):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points. Defaults to False.
        test_mode (bool): If `test_mode=True`, it will not randomly sample
            sweeps but select the nearest N frames. Defaults to False.
    """

    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 5,
                 use_dim: List[int] = [0, 1, 2, 4],
                 backend_args: Optional[dict] = None,
                 pad_empty_sweeps: bool = False,
                 remove_close: bool = False,
                 test_mode: bool = False) -> None:
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self,
                      points: Union[np.ndarray, BasePoints],
                      radius: float = 1.0) -> Union[np.ndarray, BasePoints]:
        """Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
            Updated key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                  cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if 'lidar_sweeps' not in results:
            if self.pad_empty_sweeps:
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(self._remove_close(points))
                    else:
                        sweep_points_list.append(points)
        else:
            if len(results['lidar_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['lidar_sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['lidar_sweeps']),
                    self.sweeps_num,
                    replace=False)
            for idx in choices:
                sweep = results['lidar_sweeps'][idx]
                points_sweep = self._load_points(
                    sweep['lidar_points']['lidar_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                sweep_ts = sweep['timestamp']
                lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                points_sweep[:, :
                             3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@TRANSFORMS.register_module()
class LoadMultiFrame(BaseTransform):
    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 7,
                 test_mode: bool = False,
                 backend_args = None) -> None:
        self.sweeps_num = sweeps_num
        self.test_mode = test_mode
        self.backend_args = backend_args
        self.load_dim = load_dim

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def transform(self, results: dict) -> dict:
        points = results['points']
        points.tensor = torch.cat((points.tensor, torch.zeros((points.shape[0],1))), dim=1) # 当前帧最后一维度拼接0
        points.points_dim = points.points_dim + 1
        sweep_points_list = [points]

        for idx in range(1, self.sweeps_num):
            if idx < len(results['sweeps']):
                sweep = results['sweeps'][idx]
                root_path = results['lidar_path'].split("/")[0] + '/' + results['lidar_path'].split("/")[1] + '/'
                before_lidar_file = root_path + sweep['velodyne_path']
                points_sweep = self._load_points(
                        before_lidar_file)
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                radar2_radar1 = sweep['radar2_radar1']
                points_sweep[:,:3] = (np.concatenate((points_sweep[:,:3], np.ones((points_sweep.shape[0], 1))), axis=1) @ radar2_radar1.T)[:, :3]
                points_sweep = np.concatenate((points_sweep, idx * np.ones((points_sweep.shape[0],1))), axis=1)
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

            else:
                points_sweep = np.concatenate((points.tensor[:,:-1], -idx * np.ones((points.tensor.shape[0],1))), axis=1)
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        results['points'] = points
        results['sweep_num'] = len(sweep_points_list)
        
        return results


@TRANSFORMS.register_module()
class LoadMultiFrame_allinfo(BaseTransform):
    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 7,
                 lidar_radar_path = None,
                 stacon_point_2_curframe: bool = False,
                 moncon_point_2_curframe: bool = False,
                 create_flow: bool = False):
        self.sweeps_num = sweeps_num
        self.stacon_point_2_curframe = stacon_point_2_curframe
        self.moncon_point_2_curframe = moncon_point_2_curframe
        self.create_flow = create_flow
        self.load_dim = load_dim
        self.backend_args = None
        self.track_label_path = 'data/label_track/'
        self.lidar_radar_path = lidar_radar_path

    def _load_points(self, pts_filename: str) -> np.ndarray:
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points
    
    def _load_box(self, results):
        gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
        return gt_bboxes_3d
    
    def load_track_label(self, sample_idx):
        txt_path = self.track_label_path + sample_idx + '.txt'
        value = np.loadtxt(txt_path, usecols=1)
        value_list = [float(value)] if np.ndim(value)==0 else value.tolist() 
        return value_list

    def trans_boxes(self, bboxes, trans_mat):
        '''
        Args:
            bboxes: np.ndarray [N, 7]        
            trans_mat: np.ndarray [4, 4]
        '''
        rot_mat_T = trans_mat[:3, :3].T
        rot_sin = rot_mat_T[0, 1]
        rot_cos = rot_mat_T[0, 0]
        angle = np.arctan2(rot_sin, rot_cos)
        bboxes[:, 0:3] = bboxes[:, 0:3] @ rot_mat_T + trans_mat[:3, -1]
        bboxes[:, 6] += angle
        return bboxes

    def trans_from_two_boxes(self, box1, box2):
        '''
        Args:
            box1: np.ndarray [7,]
            box2: np.ndarray [7,]
        Ret:
            t_mat: from box1 to box2
        '''
        trans = box2[:3] - box1[:3]
        ang = box2[6] - box1[6]
        rot = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1]
        ])
        t_mat = np.concatenate([rot, trans[:, np.newaxis]], axis=-1)
        t_mat = np.concatenate([t_mat, np.zeros((1, 4))], axis=0)
        t_mat[-1, -1] = 1.0
        return t_mat
        
    def transform(self, results: dict) -> dict:
        sweep_point_list = []
        now_points = results['points']
        exist_sweep_num = len(results['data_info_before_list'])

        now_track_label = self.load_track_label(results['lidar_path'][-9:-4])
        now_box = results['ann_info' ]['gt_bboxes_3d']
        assert now_box.tensor.shape[0] == len(now_track_label)

        now_points.points_dim += 1
        results['points'] = now_points.new_point( \
            np.concatenate((now_points.tensor, np.zeros((now_points.tensor.shape[0], 1))), axis=1) )

        for before_tem_id in range(1, exist_sweep_num):
            data_info_before = results['data_info_before_list'][before_tem_id]
            
            before_lidar_file = data_info_before['lidar_path']
            if self.lidar_radar_path is not None:
                before_sam_id = before_lidar_file.split('/')[-1]
                before_lidar_file = self.lidar_radar_path + before_sam_id
            points_sweep = self._load_points(
                    before_lidar_file).reshape(-1, 7)
            box_sweep = self._load_box(
                    data_info_before)
            before_track_label = self.load_track_label(
                    data_info_before['lidar_path'][-9:-4])
            
            points_sweep = points_sweep.copy()

            if self.stacon_point_2_curframe:
                radar2_radar1 = results['sweeps'][before_tem_id]['radar2_radar1']
                points_sweep[:,:3] = (np.concatenate((points_sweep[:,:3], np.ones((points_sweep.shape[0], 1))), axis=1) @ radar2_radar1.T)[:,:3]
                # points_sweep = np.concatenate((points_sweep, before_tem_id * np.ones((points_sweep.shape[0], 1))), axis=1)
            else:
                points_sweep = np.concatenate((points_sweep, before_tem_id * np.ones((points_sweep.shape[0], 1))), axis=1)

            if self.moncon_point_2_curframe:
                # 先对以前帧的box 静态补偿到当前帧
                box_sweep = self.trans_boxes(box_sweep.tensor, radar2_radar1)

                # 点进行动态补偿
                for box_idx in range(box_sweep.shape[0]):
                    now_match_box = None
                    for now_match_box_idx in range(now_box.shape[0]):
                        if int(before_track_label[box_idx]) == int(now_track_label[now_match_box_idx]):
                            now_match_box = now_box[now_match_box_idx] # before frame box 对应的 cur box
                            break
                    if now_match_box is None:
                        continue
                    sweep_point_indices = points_in_boxes_cpu(
                        torch.tensor(points_sweep[np.newaxis, :, 0:3]), 
                        torch.tensor(box_sweep[box_idx][np.newaxis, np.newaxis, :7])
                        )  # (nboxes, npoints)
                    box_sweep = np.array(box_sweep)
                    sweep_inbox_mask = (sweep_point_indices.reshape(-1) == 1)
                    sweep_point_in_box = points_sweep[sweep_inbox_mask].copy()
                    t_box_target = self.trans_from_two_boxes(box_sweep[box_idx], now_match_box.tensor.squeeze(0))
                    # motion com box
                    sweep_point_in_box[:, :3] -= box_sweep[box_idx][:3]
                    sweep_point_in_box[:, :3] = (np.concatenate((sweep_point_in_box[:,0:3], np.ones((sweep_point_in_box.shape[0], 1))), axis=1) @ t_box_target.T)[:, :3]
                    sweep_point_in_box[:, :3] += box_sweep[box_idx][:3]
                    points_sweep[sweep_inbox_mask, 0:3] = sweep_point_in_box[:, :3]
            
            points_sweep = np.concatenate((points_sweep, before_tem_id * np.ones((points_sweep.shape[0], 1))), axis = 1)
            
            points_sweep = now_points.new_point(points_sweep)
            sweep_point_list.append(points_sweep)
        
        if exist_sweep_num < self.sweeps_num:
            for before_tem_id in range(exist_sweep_num, self.sweeps_num):
                points_sweep = now_points.tensor
                points_sweep = np.concatenate((points_sweep, before_tem_id * np.ones((points_sweep.shape[0], 1))), axis = 1)
                points_sweep = now_points.new_point(points_sweep)
                sweep_point_list.append(points_sweep)

        results['sweep_point_list'] = now_points.cat(sweep_point_list)
        return results
    
@TRANSFORMS.register_module()
class PointSegClassMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = label_mapping[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NormalizePointsColor(BaseTransform):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean: List[float]) -> None:
        self.color_mean = color_mean

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        input_dict['points'] = points
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromFile(BaseTransform):

    def __init__(self,
                 coord_type: str,
                 add_abs_path = True,
                 lidar_radar_path = None,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 ) -> None:
        self.add_abs_path = add_abs_path
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.lidar_radar_path = lidar_radar_path

    def _load_points(self, pts_filename: str, result=None) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if pts_filename.endswith('txt'):
            from .dataset_astyx import Lidar
            calibration = result['calibration']
            lidar = Lidar(pts_filename, calibration['lidar'])
            return lidar.getPointCloud()
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        # if self.add_abs_path:
        #     data_root = 'ABS_PATH'
        #     key_paths = ['img_path', 'lidar_path']
        #     for key_path in key_paths:
        #         if key_path in results:
        #             results[key_path] = data_root + results[key_path]
        #     results['lidar_points']['lidar_path'] = data_root + results['lidar_points']['lidar_path']
        #     if self.lidar_radar_path is not None:
        #          self.lidar_radar_path = data_root + self.lidar_radar_path

        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path, results)

        if np.isnan(points).any():
            print('[NaN] detected in points!')
            print(pts_file_path)
            return None

        if np.isinf(points).any():
            print('[Inf] detected in points!')

        if self.lidar_radar_path is not None:
            sample_bin = pts_file_path.split('/')[-1]
            pts_file_path = self.lidar_radar_path + sample_bin
            points = self._load_points(pts_file_path)

        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        # print("Min:", np.min(points, axis=0))  # → [1.0, 0.5, -1.0]
        # print("Max:", np.max(points, axis=0))  # → [4.0, 3.5, 3.0]

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str

@TRANSFORMS.register_module()
class LoadLidar(BaseTransform):
    def __init__(self,
                 coord_type = 'LIDAR',
                 load_dim: int = 7,
                 use_dim: Union[int, List[int]] = 7,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 lidar_path = None,
                 key_name = 'lidar_point') -> None:
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.lidar_path = lidar_path
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.key_name = key_name

    def _load_points(self, pts_filename: str) -> np.ndarray:
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def transform(self, results: dict) -> dict:
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(self.lidar_path + pts_file_path.split('/')[-1])
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results[self.key_name] = points
        results['lidar_path'] = self.lidar_path
        results['sample_bin_id'] = pts_file_path.split('/')[-1]
        return results

@TRANSFORMS.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def transform(self, results: dict) -> dict:
        """Convert the type of points from ndarray to corresponding
        `point_class`.

        Args:
            results (dict): input result. The value of key `points` is a
                numpy array.

        Returns:
            dict: The processed results.
        """
        assert 'points' in results
        points = results['points']

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)

        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

@TRANSFORMS.register_module()
class create_lidar_v(BaseTransform):
    def __init__(self):
        pass
    def transform(self, results: dict) -> dict:
        lidar_point = results['lidar_point'].tensor
        radar_point = results['points'].tensor
        gt_box = results['gt_bboxes_3d'].tensor
        for gt_box_id in range(gt_box.shape[0]):
            cur_gt_box = gt_box[gt_box_id]
            lidar_in_box_mask = points_in_boxes_cpu(
                        torch.tensor(lidar_point[np.newaxis, :, 0:3]), 
                        torch.tensor(cur_gt_box[np.newaxis, np.newaxis, :7])
                        )  # (nboxes, npoints)
            lidar_inbox_mask = (lidar_in_box_mask.reshape(-1) == 1)

            radar_in_box_mask = points_in_boxes_cpu(
                        torch.tensor(radar_point[np.newaxis, :, 0:3]), 
                        torch.tensor(cur_gt_box[np.newaxis, np.newaxis, :7])
                        )  # (nboxes, npoints)
            radar_inbox_mask = (radar_in_box_mask.reshape(-1) == 1)
            radar_point_in_box = radar_point[radar_inbox_mask]

            if radar_point_in_box.shape[0] > 0:  # 确保有数据
                radar_point_in_box_vr = torch.mean(radar_point_in_box[:, 4])
                radar_point_in_box_vr_temp = torch.mean(radar_point_in_box[:, 5])
            else:  # 如果为空，返回默认值
                radar_point_in_box_vr = torch.tensor(0.0, dtype=torch.float32, device=radar_point_in_box.device)
                radar_point_in_box_vr_temp = torch.tensor(0.0, dtype=torch.float32, device=radar_point_in_box.device)
                
            lidar_point[lidar_inbox_mask, 4] = radar_point_in_box_vr
            lidar_point[~lidar_inbox_mask, 4] = 0
            lidar_point[lidar_inbox_mask, 5] = radar_point_in_box_vr_temp
            lidar_point[~lidar_inbox_mask, 5] = 0
            lidar_point[:, -1] = -1 # 标志位
        
        results['lidar_point'] = results['lidar_point'].new_point(lidar_point)
        return results

@TRANSFORMS.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - ann_info (dict)

        - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
          :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
          3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): Labels of ground truths.
          Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D ground truth bboxes.
          Only when `with_bbox` is True.
        - gt_labels (np.ndarray): Labels of ground truths.
          Only when `with_label` is True.
        - depths (np.ndarray): Only when
          `with_bbox_depth` is True.
        - centers_2d (np.ndarray): Only when
          `with_bbox_depth` is True.
        - attr_labels (np.ndarray): Attribute labels of instances.
          Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): Path of instance mask file.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): Path of semantic mask file.
      Only when `with_seg_3d` is True.
    - pts_panoptic_mask_path (str): Path of panoptic mask file.
      Only when both `with_panoptic_3d` is True.

    Added Keys:

    - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
      :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
      3D ground truth bboxes. Only when `with_bbox_3d` is True
    - gt_labels_3d (np.int64): Labels of ground truths.
      Only when `with_label_3d` is True.
    - gt_bboxes (np.float32): 2D ground truth bboxes.
      Only when `with_bbox` is True.
    - gt_labels (np.int64): Labels of ground truths.
      Only when `with_label` is True.
    - depths (np.float32): Only when
      `with_bbox_depth` is True.
    - centers_2d (np.ndarray): Only when
      `with_bbox_depth` is True.
    - attr_labels (np.int64): Attribute labels of instances.
      Only when `with_attr_label` is True.
    - pts_instance_mask (np.int64): Instance mask of each point.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask (np.int64): Semantic mask of each point.
      Only when `with_seg_3d` is True.

    Args:
        with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool): Whether to load 3D instance masks for points.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D semantic masks for points.
            Defaults to False.
        with_bbox (bool): Whether to load 2D boxes. Defaults to False.
        with_label (bool): Whether to load 2D labels. Defaults to False.
        with_mask (bool): Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
        with_panoptic_3d (bool): Whether to load 3D panoptic masks for points.
            Defaults to False.
        poly2mask (bool): Whether to convert polygon annotations to bitmasks.
            Defaults to True.
        seg_3d_dtype (str): String of dtype of 3D semantic masks.
            Defaults to 'np.int64'.
        seg_offset (int): The offset to split semantic and instance labels from
            panoptic labels. Defaults to None.
        dataset_type (str): Type of dataset used for splitting semantic and
            instance labels. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 temporal_anno: bool = False,
                 with_bbox_3d: bool = True,
                 with_label_3d: bool = True,
                 with_attr_label: bool = False,
                 with_mask_3d: bool = False,
                 with_seg_3d: bool = False,
                 with_bbox: bool = False,
                 with_label: bool = False,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 with_bbox_depth: bool = False,
                 with_panoptic_3d: bool = False,
                 poly2mask: bool = True,
                 seg_3d_dtype: str = 'np.int64',
                 seg_offset: int = None,
                 dataset_type: str = None,
                 backend_args: Optional[dict] = None,
                 pts_semantic_label_path: str=None,
                 pts_motion_label_path: str=None,
                 use_teacher = False,
                 ins_semantic_label_path: str=None,):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            backend_args=backend_args)
        self.use_teacher = use_teacher
        self.temporal_anno = temporal_anno
        self.ins_semantic_label_path = ins_semantic_label_path
        self.pts_semantic_label_path = pts_semantic_label_path
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_panoptic_3d = with_panoptic_3d
        self.seg_3d_dtype = eval(seg_3d_dtype)
        self.seg_offset = seg_offset
        self.dataset_type = dataset_type

    def _load_bboxes_3d(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results: dict) -> dict:
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['pts_instance_mask_path']

        try:
            mask_bytes = get(
                pts_instance_mask_path, backend_args=self.backend_args)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        try:
            mask_bytes = get(
                pts_semantic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes loads semantic and panoptic labels from different files.

        results['pts_semantic_mask'] = pts_semantic_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results

    def _load_semantic_seg_3d_vod(self, results):
        pts_semantic_mask_path = self.pts_semantic_label_path
        idx = results['lidar_points']['lidar_path'].split('/')[-1][:-4]
        label_file = pts_semantic_mask_path  + idx + '.bin'
        label = np.fromfile(str(label_file), dtype=np.int64).reshape(-1,)
        label = label + 1 # from -1 background---> 0 background
        pts_semantic_mask = label

        if self.temporal_anno:
            label_list = []
            sweep_num = results['sweep_num']
            label_list.append(label)
            for idx in range(1, sweep_num):
                sweep = results['sweeps'][idx]
                root_path = self.pts_semantic_label_path
                before_label_file = root_path + sweep['velodyne_path'][-9:-4] + '.bin'
                label_before = np.fromfile(str(before_label_file), dtype=np.int64).reshape(-1,)
                label_before = label_before + 1
                label_list.append(label_before)
            pts_semantic_mask = np.concatenate(label_list, axis=0)
            # assert pts_semantic_mask.shape[0] == results['points'].tensor.shape[0]

        results['pts_semantic_mask'] = pts_semantic_mask
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask

        if self.use_teacher:
            pts_semantic_mask_path = self.pts_semantic_label_path
            idx = results['lidar_points']['lidar_path'][-9:-4]
            label_file = pts_semantic_mask_path  + idx + '.bin'
            label = np.fromfile(str(label_file), dtype=np.int64).reshape(-1,)
            label = label + 1 # from -1 background---> 0 background
            pts_semantic_mask_teacher = label
            pts_semantic_mask = np.concatenate((pts_semantic_mask, pts_semantic_mask_teacher), axis=0)
            results['pts_semantic_mask'] = pts_semantic_mask
            
        return results

    def _load_semantic_ins_3d_vod(self, results):
        pts_semantic_mask_path = self.ins_semantic_label_path
        idx = results['lidar_points']['lidar_path'][-9:-4]
        label_file = pts_semantic_mask_path  + idx + '.bin'
        label = np.fromfile(str(label_file), dtype=np.int64).reshape(-1,)
        # label = label + 1 # from -1 background---> 0 background
        pts_semantic_mask = label

        if 'sweep_num' in results:
            label_list = []
            sweep_num = results['sweep_num']
            label_list.append(label)
            for idx in range(1, sweep_num):
                sweep = results['sweeps'][idx]
                root_path = self.ins_semantic_label_path
                before_label_file = root_path + sweep['velodyne_path'][-9:-4] + '.bin'
                label_before = np.fromfile(str(before_label_file), dtype=np.int64).reshape(-1,)
                # label_before = label_before + 1
                label_list.append(label_before)
            pts_semantic_mask = np.concatenate(label_list, axis=0)
            assert pts_semantic_mask.shape[0] == results['points'].tensor.shape[0]

        results['pts_instance_mask'] = pts_semantic_mask
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_semantic_mask
        return results
    
    def _load_panoptic_3d(self, results: dict) -> dict:
        """Private function to load 3D panoptic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the panoptic segmentation annotations.
        """
        pts_panoptic_mask_path = results['pts_panoptic_mask_path']

        try:
            mask_bytes = get(
                pts_panoptic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_panoptic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_panoptic_mask_path)
            pts_panoptic_mask = np.fromfile(
                pts_panoptic_mask_path, dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        elif self.dataset_type == 'nuscenes':
            pts_semantic_mask = pts_semantic_mask // self.seg_offset

        results['pts_semantic_mask'] = pts_semantic_mask

        # We can directly take panoptic labels as instance ids.
        pts_instance_mask = pts_panoptic_mask.astype(np.int64)
        results['pts_instance_mask'] = pts_instance_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            if self.dataset_type == 'vod':
                results = self._load_semantic_ins_3d_vod(results)
            else:
                results = self._load_masks_3d(results)
        if self.with_seg_3d:
            if self.dataset_type == 'vod':
                results = self._load_semantic_seg_3d_vod(results)
            else:
                results = self._load_semantic_seg_3d(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_panoptic_3d={self.with_panoptic_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        repr_str += f'{indent_str}seg_offset={self.seg_offset})'

        return repr_str


@TRANSFORMS.register_module()
class LidarDet3DInferencerLoader(BaseTransform):
    """Load point cloud in the Inferencer's pipeline.

    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, coord_type='LIDAR', **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', coord_type=coord_type, **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', coord_type=coord_type, **kwargs))
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert 'points' in single_input, "key 'points' must be in input dict"
        if isinstance(single_input['points'], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)

@TRANSFORMS.register_module()
class MonoDet3DInferencerLoader(BaseTransform):
    """Load an image from ``results['images']['CAMX']['img']``. Similar with
    :obj:`LoadImageFromFileMono3D`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['images']['CAMX']['img']``.

    Added keys:
      - img
      - box_type_3d
      - box_mode_3d

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFileMono3D', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            single_input (dict): Result dict with Webcam read image in
                ``results['images']['CAMX']['img']``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        box_type_3d, box_mode_3d = get_box_type('camera')

        if isinstance(single_input['img'], str):
            inputs = dict(
                images=dict(
                    CAM_FRONT=dict(
                        img_path=single_input['img'],
                        cam2img=single_input['cam2img'])),
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class MultiModalityDet3DInferencerLoader(BaseTransform):
    """Load point cloud and image in the Inferencer's pipeline.

    Added keys:
      - points
      - img
      - cam2img
      - lidar2cam
      - lidar2img
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, load_point_args: dict, load_img_args: dict) -> None:
        super().__init__()
        self.points_from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', **load_point_args))
        self.points_from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', **load_point_args))
        coord_type = load_point_args['coord_type']
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

        self.imgs_from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **load_img_args))
        self.imgs_from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **load_img_args))

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image, point cloud and meta
            information.
        """
        assert 'points' in single_input and 'img' in single_input, \
            "key 'points', 'img' and must be in input dict," \
            f'but got {single_input}'
        if isinstance(single_input['points'], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            points_inputs = self.points_from_ndarray(inputs)
        else:
            points_inputs = self.points_from_file(inputs)

        multi_modality_inputs = points_inputs

        box_type_3d, box_mode_3d = get_box_type('lidar')

        if isinstance(single_input['img'], str):
            inputs = dict(
                img_path=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if isinstance(single_input['img'], np.ndarray):
            imgs_inputs = self.imgs_from_ndarray(inputs)
        else:
            imgs_inputs = self.imgs_from_file(inputs)

        multi_modality_inputs.update(imgs_inputs)

        return multi_modality_inputs
