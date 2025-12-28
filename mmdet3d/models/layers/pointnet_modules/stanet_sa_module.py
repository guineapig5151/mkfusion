# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange,repeat
from mmdet3d.models.layers import PAConv
from mmdet3d.utils import ConfigType
from .builder import SA_MODULES
from .stanet_tf_encoder import TransformerEncoder, Set_ViT

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) # B, N, n
    view_shape[1:] = [1] * (len(view_shape) - 1) # B,1,1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1 # 1, N, n
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # B,N,n
    new_points = points[batch_indices, idx.long(), :]
    return new_points

def rv_fps(xyz_feat,k, npoint):
    """
    Input:
        xyz: spatical information of all points, [B,N,6]
    """
    # 计算信息丰富度
    B, N = xyz_feat.shape
    info_weight = xyz_feat.argsort(dim=-1) # 升序

    npoint_frac = npoint//k # 均匀分割
    n_frac = N//k
    idxs_list = []
    
    for i in range(k):
        
        if i == k-1:
            sampling_n = npoint-i*npoint_frac
            idxs = torch.randint(i*n_frac,N,(sampling_n,))
        else:
            sampling_n = npoint_frac
            idxs = torch.randint(i*n_frac,(i+1)*n_frac,(sampling_n,))
        idxs_list.append(idxs)
    
    filtered_idx = [info_weight[:,idxs_list[i]] for i in range(k)]

    idxs = torch.cat(tuple(filtered_idx),1)
    return idxs

class RANet(nn.Module):
    """
    输入: 聚合了邻域的特征 B,C,M,N, 以及点坐标
    k bin的个数
    """
    def __init__(self,map_k, dataset) -> None:
        super().__init__()   
        self.k =map_k
        self.dataset = dataset
        self.convs = None
        self.kernel = self.k - 3
        # Optional normalization for RA features (RCS / v_r); default False keeps behavior unchanged
        self.enable_ra_norm = False
        # Optional global normalization moments: dict(rcs_mean, rcs_std, vr_mean, vr_std)
        self.ra_norm_moments = None
        if self.k > 4:
            self.convs = nn.Sequential(
                    nn.Conv2d(1,32,self.kernel),
                    # nn.BatchNorm2d(32),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(),
                    #nn.Dropout(0.2),

                    nn.Conv2d(32,64,4),# 4,4
                    # nn.BatchNorm2d(64),
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                )
        else:
            self.convs = nn.Sequential(
                    nn.Conv2d(3,32,1),
                    # nn.BatchNorm2d(32),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(),
                    #nn.Dropout(0.2),

                    nn.Conv2d(32,64,4),# 4,4
                    # nn.BatchNorm2d(64),
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                )

    def forward(self,groups_xy):# B,M,n,C
        B,M,_,_ = groups_xy.shape
        
        groups_ra = self.generate_ramap(groups_xy) #BM,1,k,k
        #print(f'{groups_ra.shape}')
        groups_ra = self.convs(groups_ra)# BM,64,1,1
        return groups_ra.squeeze().reshape(B,M,-1)


    def generate_ramap(self,group_xy):
        # B,M,n,C
        # VoD: [x, y, z, RCS, v_r, v_r_compensated, time]
        # TJ4D: [X, Y, Z, V_r, Range, Power, Alpha, Beta, Vrc]
        B,M,n,C = group_xy.shape
        device = group_xy.device
        if self.dataset == 'VoD':
            rcs_grid = group_xy[...,3]
            vr_grid = group_xy[...,5]
        elif self.dataset == 'TJ4D':
            rcs_grid = group_xy[...,5]
            vr_grid = group_xy[...,8]
        else:
            raise NotImplementedError

        # Optional normalization to improve robustness across scenes
        if self.enable_ra_norm:
            eps = 1e-6
            if isinstance(self.ra_norm_moments, dict):
                r_mean = torch.as_tensor(self.ra_norm_moments.get('rcs_mean', 0.0), device=device, dtype=group_xy.dtype)
                r_std  = torch.as_tensor(self.ra_norm_moments.get('rcs_std', 1.0),  device=device, dtype=group_xy.dtype)
                v_mean = torch.as_tensor(self.ra_norm_moments.get('vr_mean', 0.0),  device=device, dtype=group_xy.dtype)
                v_std  = torch.as_tensor(self.ra_norm_moments.get('vr_std', 1.0),   device=device, dtype=group_xy.dtype)
                rcs_grid = (rcs_grid - r_mean) / (r_std + eps)
                vr_grid  = (vr_grid  - v_mean) / (v_std + eps)
            else:
                # per-neighborhood (instance) statistics
                rcs_mean = rcs_grid.mean(dim=-1, keepdim=True)
                rcs_std = rcs_grid.std(dim=-1, keepdim=True)
                rcs_grid = (rcs_grid - rcs_mean) / (rcs_std + eps)

                vr_mean = vr_grid.mean(dim=-1, keepdim=True)
                vr_std = vr_grid.std(dim=-1, keepdim=True)
                vr_grid = (vr_grid - vr_mean) / (vr_std + eps)
        x = group_xy[...,0]
        y = group_xy[...,1]
        _range = torch.hypot(x, y)
        _azimuth = torch.atan2(y, x)
        map_range_low,map_range_high = torch.min(_range,-1)[0] ,torch.max(_range,-1)[0].to(device)
        map_azimuth_low,map_azimuth_high = torch.min(_azimuth,-1)[0] ,torch.max(_azimuth,-1)[0].to(device)
        
        unit_range = (map_range_high - map_range_low)/self.k # B,M
        unit_azimuth = (map_azimuth_high - map_azimuth_low)/self.k #0做了除数
        
        unit_range.to(device)
        unit_azimuth.to(device)
        unit_range[unit_range==0] = 1
        unit_azimuth[unit_azimuth==0] = 1
        
        range_idx = torch.div((_range-map_range_low.unsqueeze(-1)),unit_range.unsqueeze(-1),rounding_mode='floor').long().to(device) #B,M,n
        azimuth_idx = torch.div((_azimuth-map_azimuth_low.unsqueeze(-1)),unit_azimuth.unsqueeze(-1),rounding_mode='floor').long().to(device)# B, M, n

        # 处理边界值
        range_idx[range_idx==self.k] -= 1
        azimuth_idx[azimuth_idx==self.k] -= 1
        ra_map =torch.zeros(B,M,3,self.k,self.k).to(device) # B,M,k,k
        batch_idx = torch.arange(B, dtype=torch.long).view(B,1,1,1,1).repeat(1,M,1,1,1)
        points_idx = torch.arange(M, dtype=torch.long).view(1,M,1,1,1).repeat(B,1,1,1,1)
        
        for i in range(range_idx.shape[2]):
            #print(ra_map[batch_idx,points_idx,0,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)].shape)
            ra_map[batch_idx,points_idx,0,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)] += 1
            #print(f'rsc_grid {rcs_grid[...,i].shape},{ra_map[batch_idx,points_idx,1,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)].shape}')
            
            ra_map[batch_idx,points_idx,1,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)] = torch.max(
                ra_map[batch_idx,points_idx,1,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)],
                rcs_grid[...,i].reshape(B,M,1,1,1))
            
            ra_map[batch_idx,points_idx,2,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)] = torch.max(
                ra_map[batch_idx,points_idx,2,range_idx[:,:,i].reshape(B,M,1,1,1),azimuth_idx[:,:,i].reshape(B,M,1,1,1)],
                vr_grid[...,i].reshape(B,M,1,1,1))
            
        ra_map = ra_map.reshape(B*M,3,self.k,self.k)
        return ra_map.float()

class mlps(nn.Module):
    def __init__(self,in_dim,out_dims, n = None) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        #self.dp = nn.Dropout(0.2)
        last_channel = in_dim
        for i in range(len(out_dims)):
            out_channel = out_dims[i]
            self.convs.append(nn.Conv1d(last_channel,out_channel,1))
            # self.bns.append(nn.BatchNorm1d(out_channel))
            self.bns.append(nn.GroupNorm(8, out_channel))
            last_channel = out_channel    

    def forward(self,points):
        assert len(points.shape) == 3
        for i in range(len(self.convs)):
            conv = self.convs[i]
            bn = self.bns[i]
            points = F.relu(bn(conv(points)))

        return points

class NeighborGroup(nn.Module):
    def __init__(self, n_neighbor):
        super().__init__()
        self.n_neighbor = n_neighbor

    def forward(self, xyz, new_xyz, features):
        # grouped_results may contain:
        # - grouped_features: (B, C, num_point, nsample)
        # - grouped_xyz: (B, 3, num_point, nsample)
        # - grouped_idx: (B, num_point, nsample)
        dist_map = self.mearsure_dist(xyz,new_xyz,dist='l2')
        similar_value, similar_idxs = torch.sort(dist_map, dim=-1)[0], torch.sort(dist_map, dim=-1)[1]
        # take top-K neighbors; if available neighbors < K, pad last index/value to keep shape stable
        similar_value = similar_value[..., :self.n_neighbor]
        similar_idxs = similar_idxs[..., :self.n_neighbor]
        if similar_value.shape[-1] < self.n_neighbor:
            pad_k = self.n_neighbor - similar_value.shape[-1]
            last_val = similar_value[..., -1:].expand(*similar_value.shape[:-1], pad_k)
            last_idx = similar_idxs[..., -1:].expand(*similar_idxs.shape[:-1], pad_k)
            similar_value = torch.cat([similar_value, last_val], dim=-1)
            similar_idxs = torch.cat([similar_idxs, last_idx], dim=-1)

        inter_neighbor_idxs = similar_idxs # B,M,n 
        neighbor_xyz = index_points(xyz, inter_neighbor_idxs)  # B M n C
        neighbor_feature = index_points(features,inter_neighbor_idxs)
        return dict(grouped_xyz=neighbor_xyz, grouped_idx=inter_neighbor_idxs, grouped_features=neighbor_feature, similar_value=similar_value)

    def mearsure_dist(self, high, low, dist):
        if dist == 'l2':
            high_xyz = high[:,:,:3] # B,N,2
            low_xyz = low[:,:,:3] # B,M,2
            
            similarity = torch.sqrt(torch.sum((low_xyz.unsqueeze(2) - high_xyz.unsqueeze(1))**2,dim=-1))
        
        elif dist == 'feature':
            high_feature = rearrange(high, 'B M C -> B 1 M C')
            low_feature = rearrange(low, 'B K C -> B K 1 C')
            
            feature_similarities = F.cosine_similarity(high_feature,low_feature,dim=-1)
            similarity = 1-feature_similarities
            
        else:
            raise NotImplementedError
        return similarity

@SA_MODULES.register_module()
class STANetSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (List[float]): List of radius in each ball query.
        sample_nums (List[int]): Number of samples in each ball query.
        mlp_channels (List[List[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (List[str]): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS']. Defaults to ['D-FPS'].

            - F-FPS: using feature distances for FPS.
            - D-FPS: using Euclidean distances of points for FPS.
            - FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (List[int]): Range of points to apply FPS.
            Defaults to [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        use_xyz (bool): Whether to use xyz. Defaults to True.
        pool_mod (str): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Defaults to False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz
            in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx
            in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point: int,
                 n_neighbor: int,
                 tf_input_dim: int,
                 tf_hidden_size: int,
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 sample_model: int = 6,
                 dataset: str = 'VoD',
                 **kwargs) -> None:
        super(STANetSAModule, self).__init__()

        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)


        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list
        self.sample_model = sample_model
        self.dataset = dataset

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        self.ra_map = RANet(4, self.dataset)
        # optional normalization cfg: dict(enable: bool, moments: dict)
        ra_norm_cfg = kwargs.get('ra_norm_cfg', None)
        if isinstance(ra_norm_cfg, dict):
            self.ra_map.enable_ra_norm = bool(ra_norm_cfg.get('enable', False))
            self.ra_map.ra_norm_moments = ra_norm_cfg.get('moments', None)
        self.n_neighbor = n_neighbor
        self.grouper = NeighborGroup(self.n_neighbor)

        tf_config = Set_ViT(vocab_size=tf_input_dim+8, radar_dim=7, num_hidden_layers=4, neighbor_dim=self.n_neighbor, hidden_size=tf_hidden_size)
        self.tf = TransformerEncoder(tf_config)
        self.mlp = mlps(tf_input_dim+128, [tf_input_dim+128, tf_input_dim+128])

    def measure_neighbors(self,high_xyz,low_xyz):
        """
        Args:
            high_xyz: B, M, 7
            low_xyz: B, N, 7
        Return:
            dist_map: B N M
        """
        chosed_info = [0,1]
        
        high_xyz = high_xyz[:,:,:3] # B,N,2
        low_xyz = low_xyz[:,:,:3] # B,M,2
        
        dist_map = torch.sum((low_xyz.unsqueeze(2) - high_xyz.unsqueeze(1))**2,dim=-1)

        return torch.sqrt(dist_map)

    def _sample_points(self, points_xyz: Tensor, features: Tensor,
                       indices: Tensor, target_xyz: Tensor) -> Tuple[Tensor]:
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tuple[Tensor]:

            - new_xyz: (B, num_point, 3) Sampled xyz coordinates of points.
            - indices: (B, num_point) Sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        return new_xyz, indices
    
    def _sample_points_2(self, points_xyz, features, sample_model):
        xyz = points_xyz
        model = sample_model
        npoint = self.num_point[0]
        device = xyz.device
        B, N, C = xyz.shape
        assert model > 0
        # if model == 0:
        #     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        #     distance = torch.ones(B, N).to(device) * 1e10
        #     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        #     batch_indices = torch.arange(B, dtype=torch.long).to(device)
        #     for i in range(npoint):
        #         centroids[:, i] = farthest
        #         #TODO 维度问题
        #         centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        #         dist = torch.sum((xyz - centroid) ** 2, -1)
        #         mask = dist < distance
        #         distance[mask] = dist[mask]
        #         farthest = torch.max(distance, -1)[1]
        #     return centroids
        
        if model > 0:
            # VoD: [x, y, z, RCS, v_r, v_r_compensated, time]
            # TJ4D: [X, Y, Z, V_r, Range, Power, Alpha, Beta, Vrc]
            k = int(model)
            if self.dataset == 'VoD':
                each_rcs = xyz[:,:,3]
                each_vr = xyz[:,:,5]
            elif self.dataset == 'TJ4D':
                each_rcs = xyz[:,:,5]
                each_vr = xyz[:,:,8]
            else:
                raise NotImplementedError
            
            rcs_idx = rv_fps(each_rcs,k,npoint//2)
            vr_idx = rv_fps(each_vr,k,npoint-npoint//2)
            indices = torch.cat((rcs_idx,vr_idx),1)
            return index_points(xyz, indices), index_points(features, indices), indices

    def _pool_features(self, features: Tensor) -> Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (Tensor): (B, C, N, K) Features of locally grouped
                points before pooling.

        Returns:
            Tensor: (B, C, N) Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()


    def forward(
        self,
        sa_index,
        points_xyz: Tensor,
        features: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        target_xyz: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) Features of each point.
                Defaults to None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Defaults to None.
            target_xyz (Tensor, optional): (B, M, 3) New coords of the outputs.
                Defaults to None.

        Returns:
            Tuple[Tensor]:

                - new_xyz: (B, M, 3) Where M is the number of points.
                  New features xyz.
                - new_features: (B, M, sum_k(mlps[k][-1])) Where M is the
                  number of points. New feature descriptors.
                - indices: (B, M) Where M is the number of points.
                  Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        if self.sample_model == 0:
            new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                                target_xyz)
            xyz_feature = index_points(features, indices)
        else:
            new_xyz, xyz_feature, indices = self._sample_points_2(points_xyz, features, self.sample_model)
        # grouped_results may contain:
        # - grouped_features: (B, C, num_point, nsample)
        # - grouped_xyz: (B, 3, num_point, nsample)
        # - grouped_idx: (B, num_point, nsample)
        grouped_results = self.grouper(points_xyz, new_xyz, features)
        neighbor_xyz = grouped_results['grouped_xyz']
        ra_feature = self.ra_map(neighbor_xyz)
        similar_value = grouped_results['similar_value']
        # Build a dataset-aware 7-dim relative token to match
        # Set_ViT(vocab_size=tf_input_dim+8, radar_dim=7). For VoD the
        # input has exactly 7 dims -> keep 0..6; for TJ4D (9 dims), we
        # pick a semantically aligned subset [X,Y,Z,V_r,Range,Power,Vrc].
        if self.dataset == 'VoD':
            sel = [0, 1, 2, 3, 4, 5, 6]
        elif self.dataset == 'TJ4D':
            sel = [0, 1, 2, 3, 4, 5, 8]
        else:
            raise NotImplementedError(f'Unknown dataset {self.dataset} for STANetSAModule')

        rel_new = new_xyz[..., sel]
        rel_neigh = neighbor_xyz[..., sel]
        relative_xyz = rearrange(rel_new, 'b m c -> b m 1 c') - rel_neigh  # B,M,n,7
        similar_value = similar_value[...,:self.n_neighbor].unsqueeze(-1)
        grouped_results['grouped_features'] = torch.cat([grouped_results['grouped_features'],similar_value,relative_xyz],dim=-1)
        # make cls_token shape (B*M, radar_dim=7, neighbor_dim) and pad if n<K
        relative_xyz = rearrange(relative_xyz,'b m n c -> (b m) c n')
        if relative_xyz.shape[-1] < self.n_neighbor:
            pad_k = self.n_neighbor - relative_xyz.shape[-1]
            relative_xyz = F.pad(relative_xyz, (0, pad_k), value=0.0)
        cls_token = relative_xyz
        neighbor_time = neighbor_xyz.new_zeros(neighbor_xyz.shape[:3])
        neighbor_time = torch.arange(self.n_neighbor).reshape(1,1,self.n_neighbor).repeat(neighbor_xyz.shape[0],neighbor_xyz.shape[1],1).to(neighbor_xyz.device)
        # (B, mlp[-1], num_point, nsample)
        neigh_feat, cls_feature = self.tf(grouped_results['grouped_features'],
                                    neighbor_time,
                                    mask=None,
                                    cls_token=cls_token)
        cls_feature = rearrange(cls_feature, '(b M) c -> b M c', M=self.num_point[0])
        neigh_feat = rearrange(neigh_feat[:,0], '(b M) c -> b M c', M=self.num_point[0])

        if sa_index == 0:
            feature = torch.cat([ra_feature,cls_feature,xyz_feature],dim=-1)
        else:
            feature = torch.cat([ra_feature,cls_feature],dim=-1)
        new_features = self.mlp(feature.permute(0,2,1)).permute(0,2,1)
        # (B, mlp[-1], num_point)

        return new_xyz, new_features, cls_feature, indices
