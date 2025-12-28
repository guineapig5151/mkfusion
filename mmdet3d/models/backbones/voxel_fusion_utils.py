# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type,
                                        points_cam2img, points_img2cam)
from functools import partial
from mmdet3d.structures.points import get_points_type

from spconv.pytorch import SparseConvTensor
import torch
import torch.nn as nn

class sparse_cat_fuse(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, i=0):
        super().__init__()

    def get_unique(self, features_cat, indices_cat, spatial_shape, batch_size):
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        return x_out

    def forward(self, teacher: SparseConvTensor, student: SparseConvTensor):
        bs = int(teacher.indices[-1, 0]) + 1
        return_indices = teacher.indices
        mask_indices = student.indices
        return_indices_list = []
        return_feat_list = []

        for i in range(bs):
            return_indices_cur = return_indices[return_indices[:, 0] == i][:,1:]
            mask_indices_cur = mask_indices[mask_indices[:, 0] == i][:,1:]
            # 对 return_indices 和 mask_indices 每行哈希以简化比较

            hash_a = torch.sum(return_indices_cur * torch.tensor([1025 , 1025 * 1025, 1025 * 1025 * 1025]).cuda(), dim=1)
            hash_b = torch.sum(mask_indices_cur * torch.tensor([1025 , 1025 * 1025, 1025 * 1025 * 1025]).cuda(), dim=1)
            # 检查 hash_a 中的元素是否存在于 hash_b
            mask = torch.isin(hash_a, hash_b)
            mask_test = torch.isin(hash_b, hash_a)
            # 取出 student 中的匹配行
            use_return_indice = return_indices[return_indices[:, 0] == i][mask]
            assert use_return_indice.shape[0] == mask_indices_cur.shape[0]
            use_return_feat = teacher.features[return_indices[:, 0] == i][mask]
            return_indices_list.append(use_return_indice)
            return_feat_list.append(use_return_feat)
        indice = torch.cat(return_indices_list)
        feat = torch.cat(return_feat_list)

        assert torch.sum(indice - mask_indices) == 0
        return SparseConvTensor(
            feat, indice, student.spatial_shape, student.batch_size
        )

def apply_3d_transformation(pcd: Tensor,
                            coord_type: str,
                            img_meta: dict,
                            reverse: bool = False) -> Tensor:
    """Apply transformation to input point cloud.

    Args:
        pcd (Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not. Defaults to False.

    Note:
        The elements in img_meta['transformation_3d_flow']:

            - "T" stands for translation;
            - "S" stands for scale;
            - "R" stands for rotation;
            - "HF" stands for horizontal flip;
            - "VF" stands for vertical flip.

    Returns:
        Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    pcd_rotate_mat = (
        torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        if 'pcd_rotation' in img_meta else torch.eye(
            3, dtype=dtype, device=device))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in img_meta else torch.zeros(
            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord

def apply_lidar_aug(points: Tensor,
                    coord_type: str,
                    img_meta: dict,
                    reverse: bool = True) -> Tuple[Tensor]:
    return apply_3d_transformation(
    points, coord_type, img_meta, reverse=reverse) # true

def apply_img_aug(pts_2d: Tensor,
                  img_meta: dict) -> Tensor:
    img_scale_factor = (
        pts_2d.new_tensor(img_meta['scale_factor'][:2])
        if 'scale_factor' in img_meta.keys() else 1)
    img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
    img_crop_offset = (
        pts_2d.new_tensor(img_meta['img_crop_offset'])
        if 'img_crop_offset' in img_meta.keys() else 0)
    img_shape=img_meta['img_shape'][:2]
    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    # img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    # img_coors -= img_crop_offset
    pts_2d[:, 0:2] = pts_2d[:, 0:2] * img_scale_factor - img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    # coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        # coor_x = ori_w - coor_x
        # img_coors[:, 0] = ori_w - img_coors[:, 0]
        pts_2d[:, 0] = ori_w - pts_2d[:, 0]
    #return coor_x, coor_y
    return pts_2d

def proj_to_img(points: Tensor,
                img_meta: dict,
                proj_mat: Tensor,
                coord_type: str,
                inv_lidar_aug: bool = True,
                inv_img_aug: bool = True,
                with_depth: bool = False,
                normalize: bool = False):
    if inv_lidar_aug:
        # apply transformation based on info in img_meta
        points = apply_lidar_aug(
            points, coord_type, img_meta, reverse=True)

    # project points to image coordinate
    pts_2d = points_cam2img(points, proj_mat, with_depth=with_depth)

    if inv_img_aug:
        pts_2d = apply_img_aug(pts_2d, img_meta)

    if normalize:
        img_pad_shape=img_meta['input_shape'][:2]
        h, w = img_pad_shape
        pts_2d[:, 0] = pts_2d[:, 0] / w
        pts_2d[:, 1] = pts_2d[:, 1] / h
    return pts_2d
