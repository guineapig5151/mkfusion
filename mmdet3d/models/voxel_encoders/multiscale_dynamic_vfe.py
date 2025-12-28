import numpy as np
import torch
import torch.nn as nn

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS

class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated
        
@MODELS.register_module()
class MultiScaleDynamicVFE(nn.Module):
    def __init__(self, 
                 model_cfg, 
                 num_point_features, 
                 point_cloud_range, 
                 with_pre_norm = True,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 **kwargs):
        super().__init__()
        self.num_point_features = num_point_features
        ms_voxel_sizes = model_cfg.MS_VOXEL_SIZES
        point_cloud_range = torch.tensor(point_cloud_range).cuda()
        grid_size = [(point_cloud_range[3:6] - point_cloud_range[0:3]) / torch.tensor(voxel_size).cuda() for voxel_size in ms_voxel_sizes]
        grid_size = [torch.round(gs) for gs in grid_size]
        voxel_size = ms_voxel_sizes

        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, num_filters[0])[1]
        else:
            self.pre_norm = None

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        voxel_size_list = voxel_size
        grid_size_list = grid_size
        self.grid_size = grid_size_list
        self.voxel_size = [torch.tensor(voxel_size).cuda() for voxel_size in ms_voxel_sizes]
        self.point_cloud_range = point_cloud_range

        self.voxel_x = [voxel_size[0] for voxel_size in voxel_size_list]
        self.voxel_y = [voxel_size[1] for voxel_size in voxel_size_list]
        self.voxel_z = [voxel_size[2] for voxel_size in voxel_size_list]
        self.x_offset = [voxel_x / 2 + point_cloud_range[0] for voxel_x in self.voxel_x]
        self.y_offset = [voxel_y / 2 + point_cloud_range[1] for voxel_y in self.voxel_y]
        self.z_offset = [voxel_z / 2 + point_cloud_range[2] for voxel_z in self.voxel_z]

        self.scale_xyz = [grid_size[0] * grid_size[1] * grid_size[2] for grid_size in grid_size_list]
        self.scale_yz = [grid_size[1] * grid_size[2] for grid_size in grid_size_list]
        self.scale_z = [grid_size[2] for grid_size in grid_size_list]

        # A workaround, as OpenPCDet only optimize leaf module
        self.scale_embeddings = nn.Embedding(2, out_filters, _weight=torch.zeros(2, out_filters))
        self.use_scale_embed = self.model_cfg.get('USE_SCALE_EMBED', True)

    def get_output_feature_dim(self):
        return self.num_point_features

    def filter_base_coords(self, target_coords, source_coords):
        """
        从target_coords中挑出里面source_coords对应位置, 假设source_coords是target_coords子集
        """
        
        # 将三元组转换为单个整数表示，以便使用排序和搜索 b(2位)z(2位)y(4位)x(4位)
        def hash_triplets(triplets):
            # b x y z
            return triplets[:, 0] * self.scale_xyz[0] + triplets[:, 3] * self.scale_yz[0] + triplets[:, 2] * self.scale_z[0] + triplets[:, 1]

        # 哈希化三元组
        hashed_tensor1 = hash_triplets(source_coords)
        hashed_tensor2 = hash_triplets(target_coords)

        # 对tensor2进行排序
        sorted_hashed_tensor2, sorted_indices = torch.sort(hashed_tensor2)

        # 使用torch.searchsorted在排序后的tensor2中找到tensor1的位置
        positions_in_sorted = torch.searchsorted(sorted_hashed_tensor2, hashed_tensor1)

        # 将排序后的索引映射回原始索引
        positions = sorted_indices[positions_in_sorted]
        return positions

    def forward(self,
                features_input = None, 
                coors = None,
                data_sample = None,
                **kwargs):
        
    # def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        base_voxel_center = self.voxel_size[0] / 2.0
        voxel_coords_lvl = []
        voxel_features_lvl = []
        for i in range(len(self.voxel_size)):
            points: torch.Tensor = features_input # (batch_idx, x, y, z, i, e)
            if i > 0:
                ratio_xyz = self.voxel_size[i] / self.voxel_size[0]
                assert (ratio_xyz == ratio_xyz.floor()).all() 
                ratio_xyz = ratio_xyz.int()
                cur_voxel_center = self.voxel_size[i] / 2.0
                # ratio只能是2的幂次
                assert ((ratio_xyz > 0) & ((ratio_xyz & (ratio_xyz - 1)) == 0)).all()

                point_coords = []
                for ix in range(ratio_xyz[0].item()):
                    for iy in range(ratio_xyz[1].item()):
                        for iz in range(ratio_xyz[2].item()):
                            # base_voxel_center small coord center
                            delta_voxel_center = (cur_voxel_center - base_voxel_center)
                            delta_voxel_center[0] -= ix*self.voxel_size[0][0]
                            delta_voxel_center[1] -= iy*self.voxel_size[0][1]
                            delta_voxel_center[2] -= iz*self.voxel_size[0][2]
                            point_coords_single = torch.floor((points[:, 1:4] + delta_voxel_center - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()
                            point_coords_single[:, 0] = point_coords_single[:, 0] * ratio_xyz[0] + ix
                            point_coords_single[:, 1] = point_coords_single[:, 1] * ratio_xyz[1] + iy
                            point_coords_single[:, 2] = point_coords_single[:, 2] * ratio_xyz[2] + iz
                            point_coords.append(point_coords_single)

                point_coords = torch.cat(point_coords, dim=0)
                points = points.repeat([torch.prod(ratio_xyz), 1])
            else:
                point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()

            # mask = ((point_coords >= 0) & (point_coords < self.grid_size[0])).all(dim=1)
            # points = points[mask]
            # point_coords = point_coords[mask]

            merge_coords = points[:, 0].int() * self.scale_xyz[0] + \
                            point_coords[:, 0] * self.scale_yz[0] + \
                            point_coords[:, 1] * self.scale_z[0] + \
                            point_coords[:, 2]
            
            if i > 0:
                base_coords_mask = torch.isin(merge_coords, base_unq_coords)
                points = points[base_coords_mask]
                point_coords = point_coords[base_coords_mask]
                merge_coords = merge_coords[base_coords_mask]
            
            points_data = points[:, 1:].contiguous()
            
            unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, sorted=True)
            unq_coords = unq_coords.int()
            
            if i == 0:
                base_unq_coords = unq_coords
                point2voxel_map = unq_inv
            voxel_coords = torch.stack((unq_coords // self.scale_xyz[0],
                                        (unq_coords % self.scale_xyz[0]) // self.scale_yz[0],
                                        (unq_coords % self.scale_yz[0]) // self.scale_z[0],
                                        unq_coords % self.scale_z[0]), dim=1)
            
            # voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
            
            points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)

            # 0-segvfe
            points_mean_xyz = points_mean[:, :3]
            points_xyz = points[:, [1, 2, 3]].contiguous()
            f_cluster = points_xyz - points_mean_xyz[unq_inv, :]
            f_center = torch.zeros_like(points_xyz)
            f_center[:, 0] = points_xyz[:, 0] - (point_coords[:, 0].to(points_xyz.dtype) * self.voxel_x[0] + self.x_offset[0])
            f_center[:, 1] = points_xyz[:, 1] - (point_coords[:, 1].to(points_xyz.dtype) * self.voxel_y[0] + self.y_offset[0])
            # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
            f_center[:, 2] = points_xyz[:, 2] - (point_coords[:, 2].to(points_xyz.dtype) * self.voxel_z[0] + self.z_offset[0])

            if self.use_absolute_xyz:
                features = [points[:, 1:], f_cluster, f_center]
            else:
                features = [points[:, 4:], f_cluster, f_center]

            if self.with_distance:
                points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)

            if self.pre_norm is not None:
                features = self.pre_norm(features)
                
            for pfn in self.pfn_layers:
                features = pfn(features, unq_inv)

            if i > 0 and self.use_scale_embed:
                features += self.scale_embeddings(torch.tensor(i-1).long().cuda())

            voxel_coords_lvl.append(voxel_coords.contiguous())
            voxel_features_lvl.append(features.contiguous())


        voxel_coords = voxel_coords_lvl[0]
        voxel_features = torch.cat(voxel_features_lvl, dim=-1)

        return voxel_features, voxel_coords.int(), point2voxel_map
