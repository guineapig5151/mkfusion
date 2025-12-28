import time
import torch

from .multiscale_dynamic_vfe import MultiScaleDynamicVFE

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

class MultiScaleDynamicVFEFast(MultiScaleDynamicVFE):

    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs)
        self.ori_offset_list = []
        self.ratio_list = []
        self.offset_list = []
        self.ratio_xyz_list = []
        base_voxel_center = self.voxel_size[0] / 2.0
        for i in range(1, len(self.voxel_size)):
            ratio_xyz = self.voxel_size[i] / self.voxel_size[0]
            assert (ratio_xyz == ratio_xyz.floor()).all() 
            ratio_xyz = ratio_xyz.int()
            self.ratio_xyz_list.append(ratio_xyz)
            cur_voxel_center = self.voxel_size[i] / 2.0
            # ratio只能是2的幂次
            assert ((ratio_xyz > 0) & ((ratio_xyz & (ratio_xyz - 1)) == 0)).all()
            delta_voxel_center_all = []
            ratio_all = []
            offset_all = []
            ratio_xyz = ratio_xyz.tolist()
            for ix in range(ratio_xyz[0]):
                for iy in range(ratio_xyz[1]):
                    for iz in range(ratio_xyz[2]):
                        delta_voxel_center = (cur_voxel_center - base_voxel_center)
                        delta_voxel_center[0] -= ix*self.voxel_size[0][0]
                        delta_voxel_center[1] -= iy*self.voxel_size[0][1]
                        delta_voxel_center[2] -= iz*self.voxel_size[0][2]
                        delta_voxel_center_all.append(delta_voxel_center)
                        ratio_all.append([ratio_xyz[0], ratio_xyz[1], ratio_xyz[2]])
                        offset_all.append([ix, iy, iz])
            self.ori_offset_list.append(delta_voxel_center_all)
            self.ratio_list.append(ratio_all)
            self.offset_list.append(offset_all)
        self.ori_offset_list = [None, *[torch.stack(item) for item in self.ori_offset_list]]
        self.offset_list = [None, *[torch.tensor(item).int().cuda() for item in self.offset_list]]
        self.ratio_list = [None, *[torch.tensor(item).int().cuda() for item in self.ratio_list]]
        self.ratio_xyz_list = [None, *self.ratio_xyz_list]

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        
        # # debug
        base_voxel_center = self.voxel_size[0] / 2.0
        voxel_coords_lvl = []
        voxel_features_lvl = []
        #t1_ = time.perf_counter()
        for i in range(len(self.voxel_size)):
            points: torch.Tensor = batch_dict['points'] # (batch_idx, x, y, z, i, e)
            if i > 0:
                # [N, C] -> [ox*oy*oz, N, C]
                #t1 = time.perf_counter()
                points = points[None, :, :].repeat([torch.prod(self.ratio_xyz_list[i]), 1, 1])
                point_coords = torch.floor((points[..., 1:4] + self.ori_offset_list[i].unsqueeze(1) - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()
                point_coords = point_coords * self.ratio_list[i].unsqueeze(1) + self.offset_list[i].unsqueeze(1)
                point_coords = point_coords.view((-1, point_coords.shape[-1]))
                points = points.view((-1, points.shape[-1]))
                #t2 = time.perf_counter()
                #print(t2-t1)

                sanity_check = False
                if sanity_check:
                    t1 = time.perf_counter()
                    points_ori = batch_dict['points']
                    ratio_xyz = self.voxel_size[i] / self.voxel_size[0]
                    assert (ratio_xyz == ratio_xyz.floor()).all() 
                    ratio_xyz = ratio_xyz.int()
                    cur_voxel_center = self.voxel_size[i] / 2.0
                    # ratio只能是2的幂次
                    assert ((ratio_xyz > 0) & ((ratio_xyz & (ratio_xyz - 1)) == 0)).all()
                    point_coords_slow = []
                    for ix in range(ratio_xyz[0].item()):
                        for iy in range(ratio_xyz[1].item()):
                            for iz in range(ratio_xyz[2].item()):
                                delta_voxel_center = (cur_voxel_center - base_voxel_center)
                                delta_voxel_center[0] -= ix*self.voxel_size[0][0]
                                delta_voxel_center[1] -= iy*self.voxel_size[0][1]
                                delta_voxel_center[2] -= iz*self.voxel_size[0][2]
                                point_coords_single = torch.floor((points_ori[:, 1:4] + delta_voxel_center - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()
                                point_coords_single[:, 0] = point_coords_single[:, 0] * ratio_xyz[0] + ix
                                point_coords_single[:, 1] = point_coords_single[:, 1] * ratio_xyz[1] + iy
                                point_coords_single[:, 2] = point_coords_single[:, 2] * ratio_xyz[2] + iz
                                point_coords_slow.append(point_coords_single)
                    point_coords_slow = torch.cat(point_coords_slow, dim=0)
                    points_slow = points_ori.repeat([torch.prod(ratio_xyz), 1])
                    t2 = time.perf_counter()
                    #print(t2-t1)
                    print((point_coords-point_coords_slow).abs().sum())
                    print((points-points_slow).abs().sum())
            else:
                point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size[i]).int()

            mask = ((point_coords >= 0) & (point_coords < self.grid_size[0])).all(dim=1)
            points = points[mask]
            point_coords = point_coords[mask]
            merge_coords = points[:, 0].int() * self.scale_xyz[0] + \
                            point_coords[:, 0] * self.scale_yz[0] + \
                            point_coords[:, 1] * self.scale_z[0] + \
                            point_coords[:, 2]
            
            if self.model_cfg.get('ONLY_KEEP_BASE_COORDS', True) and i > 0:
                base_coords_mask = torch.isin(merge_coords, base_unq_coords)
                points = points[base_coords_mask]
                point_coords = point_coords[base_coords_mask]
                merge_coords = merge_coords[base_coords_mask]
            
            points_data = points[:, 1:].contiguous()
            
            unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, sorted=True)
            unq_coords = unq_coords.int()
            if i == 0:
                base_unq_coords = unq_coords
            voxel_coords = torch.stack((unq_coords // self.scale_xyz[0],
                                        (unq_coords % self.scale_xyz[0]) // self.scale_yz[0],
                                        (unq_coords % self.scale_yz[0]) // self.scale_z[0],
                                        unq_coords % self.scale_z[0]), dim=1)
            voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
            
            points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
            if self.use_simple_mean_vfe or (i == 0):
                features = points_mean
            else:
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

                for pfn in self.pfn_layers:
                    features = pfn(features, unq_inv)

            if i > 0 and self.use_scale_embed:
                features += self.scale_embeddings(torch.tensor(i-1).long().cuda())
            voxel_coords_lvl.append(voxel_coords.contiguous())
            voxel_features_lvl.append(features.contiguous())
            # batch_dict['voxel_features'] = points_mean.contiguous()
            # batch_dict['voxel_coords'] = voxel_coords.contiguous()
        #t2_ = time.perf_counter()
        #print(t2_-t1_)
        if self.model_cfg.get('ONLY_KEEP_BASE_COORDS', True):
            batch_dict['voxel_coords'] = voxel_coords_lvl[0]
            batch_dict['voxel_features'] = torch.cat(voxel_features_lvl, dim=-1)
        return batch_dict
