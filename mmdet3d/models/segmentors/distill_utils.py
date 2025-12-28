from spconv.pytorch import SparseConvTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

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

    def bev_out_3d(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices # xyz
        spatial_shape = x_conv.spatial_shape

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, teacher: SparseConvTensor, student: SparseConvTensor):
        teacher = self.bev_out_3d(teacher)
        bs = int(student.indices[-1, 0]) + 1
        return_indices = teacher.indices
        mask_indices = student.indices
        return_indices_list = []
        return_feat_list = []

        for i in range(bs):
            return_indices_cur = return_indices[return_indices[:, 0] == i][:,1:]
            mask_indices_cur = mask_indices[mask_indices[:, 0] == i][:,1:]
            # 对 return_indices 和 mask_indices 每行哈希以简化比较
            hash_a = torch.sum(return_indices_cur * torch.tensor([1, 1025, 1025 * 1025]).cuda(), dim=1)
            hash_b = torch.sum(mask_indices_cur * torch.tensor([1, 1025, 1025 * 1025]).cuda(), dim=1)
            # 检查 hash_a 中的元素是否存在于 hash_b
            mask = torch.isin(hash_a, hash_b)
            # 取出 student 中的匹配行
            use_return_indice = return_indices[return_indices[:, 0] == i][mask]
            assert use_return_indice.shape[0] == mask_indices_cur.shape[0]
            use_return_feat = teacher.features[return_indices[:, 0] == i][mask]
            return_indices_list.append(use_return_indice)
            return_feat_list.append(use_return_feat)
        indice = torch.cat(return_indices_list)
        feat = torch.cat(return_feat_list)

        return SparseConvTensor(
            feat, indice, teacher.spatial_shape, teacher.batch_size
        )

class sparse_agg_fuse(nn.Module):
    def __init__(self, type = 'knn', num_p = 0):
        super().__init__()
        self.agg_type = type
        self.num_p = int(num_p)

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

    def knn_aggregation(self, a_features, a_positions, b_positions, k=8, sigma=1.0):
        Na, C = a_features.shape
        Nb, _ = b_positions.shape

        # 计算 b 到 a 的距离矩阵 (Nb, Na)
        dist_matrix = torch.cdist(b_positions, a_positions, p=2)  # (Nb, Na)

        # 取每个 b 位置最近的 k 个 a 位置点
        knn_idx = dist_matrix.topk(k=k, largest=False).indices  # (Nb, K)

        # 取最近邻的特征 (Nb, K, C)
        knn_features = a_features[knn_idx]  # (Nb, K, C)

        # 计算距离并转为权重 (Softmax 归一化)
        knn_dist = dist_matrix.gather(1, knn_idx)  # (Nb, K)
        weights = torch.exp(-knn_dist ** 2 / (2 * sigma ** 2))  # 高斯权重 (Nb, K)
        weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化 (Nb, K)

        # 计算加权平均
        aggregated_features = (knn_features * weights.unsqueeze(-1)).sum(dim=1)  # (Nb, C)

        return aggregated_features

    def bev_out_3d(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices # xyz
        spatial_shape = x_conv.spatial_shape

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, teacher: SparseConvTensor, student: SparseConvTensor):
        teacher = self.bev_out_3d(teacher)

        bs = int(student.indices[-1, 0]) + 1
        return_indices = teacher.indices
        mask_indices = student.indices
        return_indices_list = []
        return_feat_list = []

        for i in range(bs):
            
            return_indices_cur = return_indices[return_indices[:, 0] == i][:,1:]
            mask_indices_cur = mask_indices[mask_indices[:, 0] == i][:,1:]

            if self.agg_type == 'knn':
                agg_teacher_feat = self.knn_aggregation(teacher.features[return_indices[:, 0] == i], return_indices_cur.float(), mask_indices_cur.float(), k = self.num_p)

            return_indices_list.append(mask_indices[mask_indices[:, 0] == i])
            return_feat_list.append(agg_teacher_feat)
            
        indice = torch.cat(return_indices_list)
        feat = torch.cat(return_feat_list)
        return SparseConvTensor(
            feat, indice, teacher.spatial_shape, teacher.batch_size
        )


class Affinity_Loss(nn.Module):

    def __init__(self, mse = True):
        super(Affinity_Loss, self).__init__()
        self.reduction = 'mean'
        self.loss_weight = 1.0
        self.mse = mse

    def get_batch(self, sp_tensor):
        indice = sp_tensor.indices
        batch_id = indice[:,]
        feature_len = torch.max()

    def forward(self, input, target):
        feature_ditill_loss = 0.0
        batch_num = int(torch.max(input.indices[:, 0])) + 1
        for batch_id in range(batch_num):
            batch_mask = (input.indices[:, 0] == batch_id)
            input_ds_temp = input.features[batch_mask]
            target_ds_temp = target.features[batch_mask]

            C_in = input_ds_temp.shape[-1]
            B = 1
            input_ds_temp = input_ds_temp.reshape(B, C_in, -1) # B, C_in, H*W
            input_affinity = torch.bmm(input_ds_temp.permute(0, 2, 1), input_ds_temp) # (B, H*W, C_in) * (B, C_in, H*W) = (B, H*W, H*W)
            target_ds_temp = target_ds_temp.reshape(B, C_in, -1) # B, C_ta, H*W
            target_affinity = torch.bmm(target_ds_temp.permute(0, 2, 1), target_ds_temp) # (B, H*W, C_ta) * (B, C_ta, H*W) = (B, H*W, H*W)

            if self.mse:
                feature_ditill_loss += F.mse_loss(input_affinity, target_affinity, reduction=self.reduction) / B * self.loss_weight
            else:
                feature_ditill_loss += F.l1_loss(input_affinity, target_affinity, reduction=self.reduction) / B * self.loss_weight

        return feature_ditill_loss





class Quality_Focal_Loss_no_reduction(nn.Module):
    '''
    input[B,M,C] not sigmoid 
    target[B,M,C], sigmoid
    '''
    def __init__(self, beta = 2.0):

        super(Quality_Focal_Loss_no_reduction, self).__init__()
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, pos_normalizer=torch.tensor(1.0)):
        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid - target
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean') * (scale_factor.abs().pow(self.beta))
        loss /= torch.clamp(pos_normalizer, min=1.0)
        return loss
    

class RDLoss(nn.Module):
    """
    Distill_2_Loss: 计算 BEV 学生模型和教师模型之间的余弦距离损失。
    适用于 BEV 任务中的跨模态蒸馏，如 LiDAR + Camera + Radar 结合的多模态 3D 目标检测。
    
    Attributes:
        scale_factor (float): 最终损失的缩放因子，默认 4.0。
        small_weight (float): 对无效场景的损失缩小系数，默认 1e-5。
    """

    def __init__(self, scale_factor=4.0, small_weight=1e-5):
        super(RDLoss, self).__init__()
        self.scale_factor = scale_factor
        self.small_weight = small_weight

    def forward(self, input, target):
        """
        计算 BEV 特征蒸馏损失。

        Args:
            bev_s_residue_list (list of torch.Tensor): 学生模型 BEV 特征列表。
            bev_t_list (list of torch.Tensor): 教师模型 BEV 特征列表。
            img_metas (list of dict): 包含图像元数据，包括 `target_1_2`, `target_1_4`, `target_1_8`, `if_scene_useful`。

        Returns:
            torch.Tensor: 计算得到的蒸馏损失。
        """
        Distill_2_loss = 0
        batch_num = int(torch.max(input.indices[:, 0])) + 1
        for batch_id in range(batch_num):
            batch_mask = (input.indices[:, 0] == batch_id)
            input_ds_temp = input.features[batch_mask]
            target_ds_temp = target.features[batch_mask]

            # 提取有效区域的特征
            source_features = input_ds_temp.permute(1,0)
            target_features = target_ds_temp.permute(1,0)

            # 计算余弦相似度
            cos_sim = F.cosine_similarity(source_features, target_features, dim=0)
            cos_distance = 1 - cos_sim
            average_distance = torch.mean(cos_distance)
            Distill_2_loss += average_distance

        return Distill_2_loss * self.scale_factor  # 放大损失


def dense_distill(self, teacher_return, student_return, layer_name):
    choice_teacher_feat = teacher_return[layer_name]
    dense_teacher = choice_teacher_feat.dense()
    ba, channels, h, d, w = dense_teacher.shape
    dense_teacher = dense_teacher.view(ba, channels, h * d * w)
    if self.proj_teacher != None:
        dense_teacher = self.proj_teacher(dense_teacher)

    dense_student = student_return[layer_name].dense().view(ba, channels, d * w * h)
    if self.proj_student != None:
        dense_student = self.proj_student(dense_student)
        
    loss_mse = self.mse_loss(dense_teacher, dense_student) * self.distill_weight
    return loss_mse

def kl_distill(teacher_log, student_log):
    loss = F.kl_div(
        F.log_softmax(student_log, dim=1),
        F.softmax(teacher_log.detach(), dim=1),
    )
    return loss
