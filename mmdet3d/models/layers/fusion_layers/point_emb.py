import math
import torch
import torch.nn as nn

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, L=6):
        super().__init__()
        self.L = L
        self.emb_dim = L * 6

    def forward(self, xyz):
        # xyz: Tensor of shape [..., 3]
        freq_bands = 2 ** torch.arange(self.L, device=xyz.device).float() * math.pi  # [L]
        freqs = xyz[..., None, :] * freq_bands[:, None]  # [..., L, 3]
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        return torch.cat([sin, cos], dim=-1).view(*xyz.shape[:-1], -1)

class MLPEmbedding(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 # hidden_dims=[64, 64], 
                 hidden_dims=[], 
                 output_dim=32, 
                 activation=nn.ReLU):
        """
        MLP 映射编码器
        Args:
            input_dim: 输入维度（通常是 3D 坐标）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出编码的维度
            activation: 激活函数（默认 ReLU）
        """
        super().__init__()
        self.emb_dim = output_dim
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())

        # 输出层
        layers.append(nn.Linear(dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入 tensor，形状为 [B, 3] 或 [..., 3]
        Returns:
            输出编码特征，形状为 [B, output_dim] 或 [..., output_dim]
        """
        original_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])  # flatten
        out = self.mlp(x)
        return out.view(*original_shape, -1)

class LearnableEmbedding(nn.Module):
    def __init__(self, num_positions, embed_dim):
        """
        Learnable 位置编码，基于索引查表
        Args:
            num_positions: 可用的位置数量（例如 1000 个位置）
            embed_dim: 输出的编码维度
        """
        super().__init__()
        self.emb_dim = embed_dim
        self.embedding = nn.Embedding(num_positions, embed_dim)

    def forward(self, position_idx):
        """
        Args:
            position_idx: [...], 每个位置是 [0, num_positions-1] 的整数索引
        Returns:
            [..., embed_dim] 的 learnable encoding
        """
        return self.embedding(position_idx)


class PointsEmbedding(nn.Module):
    def __init__(self, emb_cfg):
        super().__init__()
        emb_type = emb_cfg.pop('type')
        self.emb_dim = emb_cfg.get('emd_dim', 3)

        if emb_type == 'fourier':
            self.points_emb = FourierPositionalEmbedding(**emb_cfg)
        elif emb_type == 'mlp':
            self.points_emb = MLPEmbedding(**emb_cfg)
        elif emb_type == 'learnable':
            raise NotImplementedError
            # self.points_emb = LearnableEmbedding(**emb_cfg)
        else:
            raise NotImplementedError
        
    
    def forward(self, points):
        return self.points_emb(points)