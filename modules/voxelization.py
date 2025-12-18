import torch
import torch.nn as nn

import modules.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')



class Voxelization_2(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        """
        features: [B, C, N]
        coords:   [B, 3, N]
        return:
            vox_feats:  [B, C, r, r, r]   # 原有
            norm_coords:[B, 3, N]         # 原有，归一化后坐标（0~1）
            vox_mask:   [B, 1, r, r, r]   # 新增，占用mask（非空=1）
        """
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            # 归一到 [0,1]（中心化 + 缩放到半径以内）
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0

        # 映射到体素索引区间 [0, r-1]
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)

        # 体素平均聚合（原有）
        vox_feats = F.avg_voxelize(features, vox_coords, self.r)

        # === 新增：占用 mask（用“全1特征”的体素平均聚合；非空体素处为1，空体素为0） ===
        B, _, N = features.shape
        ones = torch.ones((B, 1, N), dtype=features.dtype, device=features.device)
        vox_mask = F.avg_voxelize(ones, vox_coords, self.r)         # 非空体素处平均值为1
        vox_mask = (vox_mask > 0).to(features.dtype)                # [B,1,r,r,r]，显式转为0/1

        return vox_feats, norm_coords, vox_mask

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')