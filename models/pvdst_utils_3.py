import functools

import torch
import torch.nn as nn
import numpy as np
import math
from modules.shared_mlp import SharedMLP
from modules.pointnet import PointNetSAModule, PointNetAModule, PointNetFPModule
from modules.se import SE3d,GAM,SEWeightModule3D,CBAM3D ,SpatialAttention3D,ChannelAttention3D
from modules.voxelization import Voxelization,Voxelization_2
import modules.functional as MF
from modules.ops import sample_and_group, sample_and_group_with_idx
from modules.ops import knn_point
from PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet
# import pointops
from modules.ops import knn_point
import torch.nn.functional as F
# from voxel_cnn import DaliCMoudle_3,DaliCMoudle_conv
from voxel_cnn_1 import DaliCMoudle_3,DaliCMoudle_conv,DaliCMoudle_conv_4,MKCB3D

class GeometryEncoding_v2(nn.Module):
    def __init__(self, enc_channels, rel=True, abs=False, euc=True):

        super().__init__()
        self.rel = rel
        self.abs = abs
        self.euc = euc
        self.k=16
        self.d = nn.Parameter(torch.randn(size=(1, enc_channels,1,self.k)), requires_grad=True)
        self.p = nn.Parameter(torch.randn(size=(1, enc_channels,1,self.k)), requires_grad=True)
        nn.init.xavier_uniform_(self.d.data)
        nn.init.xavier_uniform_(self.p.data)

        in_channels = 3 #0
        in_channels += 3 if rel else 0
        in_channels += 3 if abs else 0
        in_channels += 1 if euc else 0
        self.ge = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, 1),
            nn.BatchNorm2d(enc_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(enc_channels, enc_channels, 1)
        )
        self.pl_f = nn.Sequential(
            nn.Conv2d(3, enc_channels, 1), #enc_channels
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1)
        )
        self.c_f = nn.Sequential(
            nn.Conv2d(4, enc_channels, 1),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1)
        )
        # self.gate = nn.Sequential(
        #     nn.Conv2d(enc_channels * 2, enc_channels, 1, bias=True),
        #     nn.BatchNorm2d(enc_channels),
        #     # nn.GroupNorm(8, enc_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(enc_channels, enc_channels, 1, bias=True)
        # )
        # for m in self.gate.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        # self.norm_c = nn.BatchNorm2d(enc_channels)
        # self.norm_p = nn.BatchNorm2d(enc_channels)
    def forward(self, neighboe_xyz, center_xyz):
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)
        enc_features_list = []
        if self.rel:
            enc_features_list.append(xyz_diff)
        if self.abs:
            enc_features_list.append(neighboe_xyz)
        if self.euc:
            enc_features_list.append(torch.norm(xyz_diff, p=2, dim=1).unsqueeze(1)) #8,1,10000,16
        enc_features = torch.cat(enc_features_list, dim=1)
        enc_features=self.c_f(enc_features)
        e_dis = torch.from_numpy(np.linalg.norm(xyz_diff.detach().cpu().numpy(), axis=1)).unsqueeze(1).cuda()
        e_dis_xy = torch.from_numpy(np.linalg.norm(xyz_diff.detach().cpu().numpy()[:, :2, :,:], axis=1)).unsqueeze(1).cuda()
        cos_theta_e = (e_dis_xy / e_dis).nan_to_num(0)
        cos_theta_a = (xyz_diff[:,1, :, :].unsqueeze(1)/ e_dis_xy).nan_to_num(0)
        # ploar_f = torch.cat([e_dis, cos_theta_a, cos_theta_e], dim=1)
        ploar_f = torch.cat([cos_theta_a, cos_theta_e], dim=1)

        ploar_f=self.pl_f(ploar_f)

        # all_feat = self.p * enc_features+self.d * ploar_f
        all_feat = enc_features+self.d * ploar_f
        # enc_features = self.norm_c(enc_features)
        # ploar_f = self.norm_p(ploar_f)

        # g = self.gate(torch.cat([enc_features, ploar_f], dim=1))  # [B,C,N,K], in (0,1)
        # all_feat = enc_features + g * (ploar_f - enc_features)

        return self.ge(all_feat)


class GeometryEncoding_v3(nn.Module):
    def __init__(self, enc_channels, rel=True, abs=False, euc=True):
        """
        enc_channels: output_channels
        rel: relative position
        abs: absolute position
        euc: euclidean distance
        """
        super().__init__()
        enc_channels =16
        self.rel = rel
        self.abs = abs
        self.euc = euc
        self.k=16
        self.d = nn.Parameter(torch.randn(size=(1, enc_channels,1,self.k)), requires_grad=True)
        self.p = nn.Parameter(torch.randn(size=(1, enc_channels,1,self.k)), requires_grad=True)
        nn.init.xavier_uniform_(self.d.data)
        nn.init.xavier_uniform_(self.p.data)
        in_channels = 3 #0
        in_channels += 3 if rel else 0
        in_channels += 3 if abs else 0
        in_channels += 1 if euc else 0
        self.ge = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, 1, bias=True),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1, bias=True)
        )
        self.pl_f = nn.Sequential(
            nn.Conv2d(5, enc_channels, 1, bias=True), #enc_channels
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1, bias=True)
        )
        self.c_f = nn.Sequential(
            nn.Conv2d(4, enc_channels, 1, bias=True),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1, bias=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(enc_channels * 2, enc_channels, 1, bias=True),
            # nn.BatchNorm2d(enc_channels),
            nn.GroupNorm(8, enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1, bias=True)
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, neighboe_xyz, center_xyz):
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)
        enc_features_list = []
        if self.rel:
            enc_features_list.append(xyz_diff)
        if self.abs:
            enc_features_list.append(neighboe_xyz)
        if self.euc:
            enc_features_list.append(torch.norm(xyz_diff, p=2, dim=1).unsqueeze(1)) #8,1,10000,16
        enc_features = torch.cat(enc_features_list, dim=1)
        enc_features=self.c_f(enc_features)
        eps = 1e-6
        r = torch.norm(xyz_diff, dim=1, keepdim=True).clamp_min(eps)  # [B,1,N,K]
        rho = torch.norm(xyz_diff[:, :2, ...], dim=1, keepdim=True).clamp_min(eps)
        r_mean = r.mean(dim=-1, keepdim=True).detach()  # [B,1,N,1]
        rho_mean = rho.mean(dim=-1, keepdim=True).detach()
        r_norm = (r / (r_mean + eps)).clamp(max=3.0)
        theta = torch.atan2(xyz_diff[:, 1, ...], xyz_diff[:, 0, ...]).unsqueeze(1)   # [B,1,N,K]
        phi = torch.atan2(xyz_diff[:, 2, ...], rho.squeeze(1)).unsqueeze(1)          # [B,1,N,K]
        ploar_f = torch.cat([
            r_norm, torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)
        ], dim=1)
        ploar_f=self.pl_f(ploar_f)
        # g = self.gate(torch.cat([enc_features, ploar_f], dim=1))  # [B,C,N,K], in (0,1)
        w = self.gate(torch.cat([enc_features, ploar_f], dim=1))  # [B,C,N,K], in (0,1)
        g = torch.sigmoid(w)#/ 1.5
        all_feat = enc_features + g * (ploar_f - enc_features)

        return self.ge(all_feat)



class GeometryEncoding(nn.Module):
    def __init__(self, enc_channels, rel=True, abs=False, euc=True):
        """
        enc_channels: output_channels
        rel: relative position
        abs: absolute position
        euc: euclidean distance
        """
        super().__init__()
        self.rel = rel
        self.abs = abs
        self.euc = euc

        in_channels = 0
        in_channels += 3 if rel else 0
        in_channels += 3 if abs else 0
        in_channels += 1 if euc else 0
        self.ge = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels, 1),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, 1)
        )

    def forward(self, neighboe_xyz, center_xyz):
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)

        enc_features_list = []
        if self.rel:
            enc_features_list.append(xyz_diff)
        if self.abs:
            enc_features_list.append(neighboe_xyz)
        if self.euc:
            enc_features_list.append(torch.norm(xyz_diff, p=2, dim=1).unsqueeze(1))


        enc_features = torch.cat(enc_features_list, dim=1)

        return self.ge(enc_features)

class GeometryEncoding_v5(nn.Module):
    def __init__(self, enc_channels, rel=True, abs=False, euc=True,
                 R_bins=2, A_bins=12, eps=1e-6):
        super().__init__()
        self.rel, self.abs, self.euc = rel, abs, euc
        self.R, self.A = R_bins, A_bins
        self.eps = eps
        C = enc_channels
        in_cart = 0
        if rel: in_cart += 3                 # Δxyz
        if abs: in_cart += 3                 # 邻域绝对 xyz
        if euc: in_cart += 1                 # ||Δ||
        self.c_f = nn.Sequential(
            nn.Conv2d(in_cart, C, 1, bias=True),   # [B,in_cart,N,K] -> [B,C,N,K]
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=True)         # [B,C,N,K] -> [B,C,N,K]
        )
        self.pl_f = nn.Sequential(
            nn.Conv2d(5, C, 1, bias=True),        # [B,5,N,K] -> [B,C,N,K]
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=True)         # [B,C,N,K] -> [B,C,N,K]
        )
        self.k=16
        self.align_c = nn.Conv2d(C, C, 1, bias=True)  # 对齐直角分支
        self.align_p = nn.Conv2d(C, C, 1, bias=True)  # 对齐极坐标分支
        # self.p = nn.Parameter(torch.randn(size=(1, enc_channels, 1, self.k)), requires_grad=True)
        # nn.init.xavier_uniform_(self.p.data)
        self.gamma   = nn.Parameter(torch.zeros(1))
        self.ge = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=True),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=True)
        )
    def _polar_feats(self, xyz_diff):
        eps = self.eps
        r   = torch.norm(xyz_diff, dim=1, keepdim=True).clamp_min(eps)     # 半径 r: [B,1,N,K]
        rho = torch.norm(xyz_diff[:, :2, ...], dim=1, keepdim=True).clamp_min(eps)  # 投影半径 ρ: [B,1,N,K]
        theta = torch.atan2(xyz_diff[:, 1, ...], xyz_diff[:, 0, ...]).unsqueeze(1)  # θ: [B,1,N,K] ∈ (-π,π]
        phi   = torch.atan2(xyz_diff[:, 2, ...], rho.squeeze(1)).unsqueeze(1)       # φ: [B,1,N,K]
        # r_max = r.max(dim=-1, keepdim=True)[0].detach()
        # r_norm = (r / (r_max + eps)).clamp(max=1.0)                # [B,1,N,K] ∈ [0,1]

        r_mean = r.mean(dim=-1, keepdim=True).detach()                      # [B,1,N,1]
        r_norm = (r / (r_mean + eps)).clamp(max=3.0) / 3.0                  # [B,1,N,K] ∈ [0,1]

        polar = torch.cat([r_norm,
                           torch.sin(theta), torch.cos(theta),
                           torch.sin(phi),   torch.cos(phi)], dim=1)        # [B,5,N,K]
        return polar

    def _cart_to_polar_indices(self, xyz_diff):
        eps = self.eps
        r   = torch.norm(xyz_diff, dim=1, keepdim=True).clamp_min(eps)      # [B,1,N,K]
        theta = torch.atan2(xyz_diff[:, 1, ...], xyz_diff[:, 0, ...]).unsqueeze(1)  # [B,1,N,K]
        r_mean = r.mean(dim=-1, keepdim=True).detach()
        r_norm = (r / (r_mean + eps)).clamp(max=3.0) / 3.0                  # [0,1]
        # r_max = r.max(dim=-1, keepdim=True)[0].detach()
        # r_norm = (r / (r_max + eps)).clamp(max=1.0)                # [B,1,N,K] ∈ [0,1]

        a_unit = (theta + math.pi) / (2 * math.pi)                        # 角度归一到 [0,1)
        r_idx_real = r_norm * (self.R - 1e-6)                                # 连续半径索引 ∈ [0,R)
        a_idx_real = a_unit * (self.A - 1e-6)                                # 连续角度索引 ∈ [0,A)
        return r_idx_real, a_idx_real

    @staticmethod
    def _bilinear_weights(idx_real, max_bin):
        idx0 = torch.floor(idx_real)                     # 下界
        idx1 = idx0 + 1.0                                # 上界
        w1   = idx_real - idx0                           # 上界权重
        w0   = 1.0 - w1                                  # 下界权重
        idx0 = idx0.long().clamp(min=0, max=max_bin - 1) # clamp 到合法格点
        idx1 = idx1.long().clamp(min=0, max=max_bin - 1)
        return idx0, idx1, w0, w1

    def _rasterize_to_polar_grid(self, X, r_idx_real, a_idx_real):
        B, C, N, K = X.shape
        R, A = self.R, self.A
        ri0, ri1, rw0, rw1 = self._bilinear_weights(r_idx_real, R)
        ai0, ai1, aw0, aw1 = self._bilinear_weights(a_idx_real, A)
        w00 = rw0 * aw0; w01 = rw0 * aw1
        w10 = rw1 * aw0; w11 = rw1 * aw1
        X_flat  = X.reshape(B, C, N * K)                                   # 展平便于 scatter
        idx_base = torch.arange(N, device=X.device).view(1,1,N,1).repeat(B,1,1,K)  # n 基址
        def _acc(ri, ai, w):
            idx = (idx_base * (R*A) + ri * A + ai).reshape(B,1,N*K)        # 合成单一索引
            val = X_flat * w.reshape(B,1,N*K)                               # 加权值
            return idx, val, w
        packs = [_acc(ri0, ai0, w00), _acc(ri0, ai1, w01),
                 _acc(ri1, ai0, w10), _acc(ri1, ai1, w11)]
        grid_feat = X.new_zeros(B, C, N * R * A)                            # 网格累加器
        grid_cnt  = X.new_zeros(B, 1, N * R * A)                            # 权重计数器

        for idx, val, w in packs:
            grid_feat.scatter_add_(2, idx.expand_as(val), val)              # 特征加权求和
            grid_cnt.scatter_add_(2, idx, w.reshape(B,1,-1))                # 权重累加

        grid_feat = grid_feat / grid_cnt.clamp_min(self.eps)                # 归一化
        return grid_feat.reshape(B, C, N, R, A)                              # [B,C,N,R,A]

    def _sample_from_polar_grid(self, grid_feat, r_idx_real, a_idx_real):
        """
        grid_feat:  [B,C,N,R,A]
        r_idx_real: [B,1,N,K]  连续半径索引 ∈ [0,R)
        a_idx_real: [B,1,N,K]  连续角度索引 ∈ [0,A)
        return:     [B,C,N,K]
        """
        B, C, N, R, A = grid_feat.shape
        ri0, ri1, rw0, rw1 = self._bilinear_weights(r_idx_real, R)
        ai0, ai1, aw0, aw1 = self._bilinear_weights(a_idx_real, A)

        def _gather(ri, ai):
            # 先在 R 维 gather：ri_s: [B,1,N,1] → 扩展为 [B,1,N,1,A]
            ri_s = ri[:, :, :, :1]  # 取一个标量，去掉 K 维
            ri_idx = ri_s.unsqueeze(-1).expand(-1, 1, -1, 1, A)  # [B,1,N,1,A]
            take_r = grid_feat.gather(
                3,  # dim=3 (R)
                ri_idx.expand(-1, C, -1, -1, -1)  # [B,C,N,1,A]
            )
            # 再在 A 维 gather：ai_s: [B,1,N,1] → 扩展为 [B,1,N,1,1]
            ai_s = ai[:, :, :, :1]  # 取一个标量，去掉 K 维
            ai_idx = ai_s.unsqueeze(-1)  # [B,1,N,1,1]
            take_ra = take_r.gather(
                4,  # dim=4 (A)
                ai_idx.expand(-1, C, -1, 1, -1)  # [B,C,N,1,1]
            )
            return take_ra.squeeze(-1).squeeze(-1)  # [B,C,N]
        # 四个角点（R×A）的双线性采样（先“抹 K”，后用权重把 K 带回去）
        g00 = _gather(ri0, ai0)
        g01 = _gather(ri0, ai1)
        g10 = _gather(ri1, ai0)
        g11 = _gather(ri1, ai1)
        # 权重（含 K 维）: [B,1,N,K]
        w00 = rw0 * aw0
        w01 = rw0 * aw1
        w10 = rw1 * aw0
        w11 = rw1 * aw1
        # 外积广播回 K：最终 [B,C,N,K]
        X_align = (g00.unsqueeze(-1) * w00 +
                   g01.unsqueeze(-1) * w01 +
                   g10.unsqueeze(-1) * w10 +
                   g11.unsqueeze(-1) * w11)
        return X_align

    def forward(self, neighboe_xyz, center_xyz):
        """
        neighboe_xyz: [B,3,N,K]  邻域坐标
        center_xyz:   [B,3,N]    中心坐标
        return:       [B,C,N,K]  融合后的几何编码
        """
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)                    # [B,3,N,K]
        feats_cart = []
        if self.rel: feats_cart.append(xyz_diff)                              # Δxyz
        if self.abs: feats_cart.append(neighboe_xyz)                          # 绝对 xyz
        if self.euc: feats_cart.append(torch.norm(xyz_diff, p=2, dim=1, keepdim=True)) # ||Δ||
        cart_in = torch.cat(feats_cart, dim=1)                                # [B,in_cart,N,K]
        C_raw = self.c_f(cart_in)                                             # [B,C,N,K]
        polar_in = self._polar_feats(xyz_diff)                                 # [B,5,N,K]
        P_raw    = self.pl_f(polar_in)                                         # [B,C,N,K]
        r_idx_real, a_idx_real = self._cart_to_polar_indices(xyz_diff)         # 连续索引
        grid_P    = self._rasterize_to_polar_grid(P_raw, r_idx_real, a_idx_real)   # P: K -> R×A
        P_aligned = self._sample_from_polar_grid(grid_P, r_idx_real, a_idx_real)   # R×A -> K
        C = self.align_c(C_raw)                                                # 对齐通道/尺度
        P = self.align_p(P_aligned)
        # fused = C + self.p * (P)
        fused = C + self.gamma* (P)

        # 输出头
        return self.ge(fused)                                                  # [B,C,N,K]



class GeometryEncodingMulti(nn.Module):
    def __init__(self,
                 enc_channels,
                 rel=True,
                 abs=False,
                 euc=True,
                 use_simple=True,
                 add_polar=True,
                 R_bins=4,
                 A_bins=8,
                 eps=1e-6):
        super().__init__()
        self.rel = rel
        self.abs = abs
        self.euc = euc
        self.use_simple = use_simple
        self.add_polar = add_polar
        self.R = R_bins
        self.A = A_bins
        self.eps = eps
        C = enc_channels
        in_cart = 0
        if rel:
            in_cart += 3       # Δxyz
        if abs:
            in_cart += 3       # 绝对 xyz
        if euc:
            in_cart += 1       # ||Δ||
        assert in_cart > 0, "At least one of (rel, abs, euc) must be True."
        # self.ge
        if self.use_simple:
            self.ge = nn.Sequential(
                nn.Conv2d(in_cart, C, 1, bias=True),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 1, bias=True)
            )
        if self.add_polar:
            self.pl_f = nn.Sequential(
                nn.Conv2d(5, C, 1, bias=True),     # [B,5,N,K] -> [B,C,N,K]
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 1, bias=True)      # [B,C,N,K] -> [B,C,N,K]
            )
            self.align_c = nn.Conv2d(C, C, 1, bias=True)
            self.align_p = nn.Conv2d(C, C, 1, bias=True)
            self.gamma = nn.Parameter(torch.zeros(1))
            # self.out_proj \
            self.c_f = nn.Sequential(
                nn.Conv2d(in_cart, C, 1),  # [B,in_cart,N,K] -> [B,C,N,K]
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 1)  # [B,C,N,K] -> [B,C,N,K]
            )
            self.ge= nn.Sequential(
                nn.Conv2d(C, C, 1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 1)
            )


        # self.out_proj = nn.Sequential(
        #     nn.Conv2d(C, C, 1),
        #     nn.BatchNorm2d(C),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(C, C, 1)
        # )
    def _polar_feats(self, xyz_diff):
        """
        xyz_diff: [B,3,N,K]
        输出:     [B,5,N,K]  (r_norm, sinθ, cosθ, sinφ, cosφ)
        """
        eps = self.eps
        r = torch.norm(xyz_diff, dim=1, keepdim=True).clamp_min(eps)        # [B,1,N,K]
        rho = torch.norm(xyz_diff[:, :2, ...], dim=1, keepdim=True).clamp_min(eps)
        theta = torch.atan2(xyz_diff[:, 1, ...], xyz_diff[:, 0, ...]).unsqueeze(1)  # [B,1,N,K]
        phi = torch.atan2(xyz_diff[:, 2, ...], rho.squeeze(1)).unsqueeze(1)         # [B,1,N,K]
        r_mean = r.mean(dim=-1, keepdim=True).detach()                     # [B,1,N,1]
        r_norm = (r / (r_mean + eps)).clamp(max=3.0) / 3.0                 # [0,1]

        polar = torch.cat([
            r_norm,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi),   torch.cos(phi)
        ], dim=1)                                                          # [B,5,N,K]
        return polar

    def _cart_to_polar_indices(self, xyz_diff):
        eps = self.eps
        r = torch.norm(xyz_diff, dim=1, keepdim=True).clamp_min(eps)       # [B,1,N,K]
        theta = torch.atan2(xyz_diff[:, 1, ...], xyz_diff[:, 0, ...]).unsqueeze(1)
        r_mean = r.mean(dim=-1, keepdim=True).detach()
        r_norm = (r / (r_mean + eps)).clamp(max=3.0) / 3.0                 # [0,1]
        a_unit = (theta + math.pi) / (2 * math.pi)
        r_idx_real = r_norm * (self.R - 1e-6)
        a_idx_real = a_unit * (self.A - 1e-6)
        return r_idx_real, a_idx_real

    @staticmethod
    def _bilinear_weights(idx_real, max_bin):
        idx0 = torch.floor(idx_real)                     # 下界
        idx1 = idx0 + 1.0                                # 上界
        w1   = idx_real - idx0                           # 上界权重
        w0   = 1.0 - w1                                  # 下界权重
        idx0 = idx0.long().clamp(min=0, max=max_bin - 1)
        idx1 = idx1.long().clamp(min=0, max=max_bin - 1)
        return idx0, idx1, w0, w1

    def _rasterize_to_polar_grid(self, X, r_idx_real, a_idx_real):
        """
        将 K 邻域特征投影到 (R, A) 极坐标网格上（每个中心点一张 R×A 网格）

        X:           [B,C,N,K]
        r_idx_real:  [B,1,N,K]
        a_idx_real:  [B,1,N,K]

        返回:
            grid_feat: [B,C,N,R,A]
        """
        B, C, N, K = X.shape
        R, A = self.R, self.A

        ri0, ri1, rw0, rw1 = self._bilinear_weights(r_idx_real, R)
        ai0, ai1, aw0, aw1 = self._bilinear_weights(a_idx_real, A)
        w00 = rw0 * aw0
        w01 = rw0 * aw1
        w10 = rw1 * aw0
        w11 = rw1 * aw1

        X_flat = X.reshape(B, C, N * K)                                  # [B,C,N*K]

        idx_base = torch.arange(N, device=X.device, dtype=torch.long).view(1, 1, N, 1)
        idx_base = idx_base.repeat(B, 1, 1, K)                           # [B,1,N,K]

        def _acc(ri, ai, w):
            idx = (idx_base * (R * A) + ri * A + ai).reshape(B, 1, N * K)   # [B,1,N*K]
            val = X_flat * w.reshape(B, 1, N * K)                           # [B,C,N*K]
            return idx, val, w

        packs = [
            _acc(ri0, ai0, w00),
            _acc(ri0, ai1, w01),
            _acc(ri1, ai0, w10),
            _acc(ri1, ai1, w11),
        ]

        grid_feat = X.new_zeros(B, C, N * R * A)
        grid_cnt  = X.new_zeros(B, 1, N * R * A)

        for idx, val, w in packs:
            grid_feat.scatter_add_(2, idx.expand_as(val), val)
            grid_cnt.scatter_add_(2, idx, w.reshape(B, 1, -1))

        grid_feat = grid_feat / grid_cnt.clamp_min(self.eps)             # 归一化
        return grid_feat.reshape(B, C, N, R, A)

    def _sample_from_polar_grid(self, grid_feat, r_idx_real, a_idx_real):
        """
        从 (R,A) 极坐标网格上对每个邻域方向做“反采样”，得到 [B,C,N,K] 的对齐特征。
        注意：这里沿 K 维做了聚合，网格是 per-center 的，方向级别由 w00...w11 再注入回来。

        grid_feat:  [B,C,N,R,A]
        r_idx_real: [B,1,N,K]
        a_idx_real: [B,1,N,K]
        """
        B, C, N, R, A = grid_feat.shape

        ri0, ri1, rw0, rw1 = self._bilinear_weights(r_idx_real, R)
        ai0, ai1, aw0, aw1 = self._bilinear_weights(a_idx_real, A)

        def _gather(ri, ai):
            ri_s = ri[:, :, :, :1]                               # [B,1,N,1]
            ri_idx = ri_s.unsqueeze(-1).expand(-1, 1, -1, 1, A)  # [B,1,N,1,A]
            take_r = grid_feat.gather(
                3,  # dim=3 对应 R
                ri_idx.expand(-1, C, -1, -1, -1)                 # [B,C,N,1,A]
            )
            ai_s = ai[:, :, :, :1]                               # [B,1,N,1]
            ai_idx = ai_s.unsqueeze(-1)                          # [B,1,N,1,1]
            take_ra = take_r.gather(
                4,  # dim=4 对应 A
                ai_idx.expand(-1, C, -1, 1, -1)                  # [B,C,N,1,1]
            )
            return take_ra.squeeze(-1).squeeze(-1)               # [B,C,N]

        g00 = _gather(ri0, ai0)  # [B,C,N]
        g01 = _gather(ri0, ai1)
        g10 = _gather(ri1, ai0)
        g11 = _gather(ri1, ai1)

        w00 = rw0 * aw0   # [B,1,N,K]
        w01 = rw0 * aw1
        w10 = rw1 * aw0
        w11 = rw1 * aw1

        X_align = (g00.unsqueeze(-1) * w00 +
                   g01.unsqueeze(-1) * w01 +
                   g10.unsqueeze(-1) * w10 +
                   g11.unsqueeze(-1) * w11)

        return X_align  # [B,C,N,K]

    def forward(self, neighboe_xyz, center_xyz, return_components=False):
        xyz_diff = neighboe_xyz - center_xyz.unsqueeze(-1)
        feats_cart = []
        if self.rel:
            feats_cart.append(xyz_diff)
        if self.abs:
            feats_cart.append(neighboe_xyz)
        if self.euc:
            feats_cart.append(torch.norm(xyz_diff, p=2, dim=1, keepdim=True))

        cart_in = torch.cat(feats_cart, dim=1)                # [B,in_cart,N,K]


        if self.add_polar:
            C_raw =self.c_f(cart_in)
            polar_in = self._polar_feats(xyz_diff)            # [B,5,N,K]
            P_raw    = self.pl_f(polar_in)                    # [B,C,N,K]
            r_idx_real, a_idx_real = self._cart_to_polar_indices(xyz_diff)
            grid_P   = self._rasterize_to_polar_grid(P_raw, r_idx_real, a_idx_real)   # [B,C,N,R,A]
            P_align  = self._sample_from_polar_grid(grid_P, r_idx_real, a_idx_real)   # [B,C,N,K]
            C_align = self.align_c(C_raw)
            P_align = self.align_p(P_align)
            F_polar = C_align + self.gamma * P_align    # [B,C,N,K]
            fused = F_polar
            out = self.ge(fused)  # [B,C,N,K]
            return out

        elif self.use_simple:
            out =self.ge(cart_in)

            return out





class NeighborEmbedding(nn.Module):
    def __init__(self, in_channels, emb_channels, form='squ', geo=True):
        """
        emb_channels: output_channels
        form: 'abs', 'squ', 'mul'
            abs: absolute difference
            squ: square difference
            mul: multiplication
        geo: geometry enconde
        """
        super().__init__()
        self.form = form
        self.geo = geo
        
        self.ne = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, emb_channels, 1),
        )

    def forward(self, key, query, geo=None):
        if self.form == 'abs':
            diff = key - query.unsqueeze(-1)
            emb_features = - torch.abs(diff)
        elif self.form == 'squ':
            diff = key - query.unsqueeze(-1)
            # emb_features = - diff * diff
            emb_features = diff * diff

        elif self.form == 'mul':
            prod = key * query.unsqueeze(-1)
            emb_features = prod
        
        if self.geo:
            emb_features += geo

        return self.ne(emb_features)

class WeakLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 4
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.radius=0.1
        self.k=16
        self.grouper=knn
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_uni_spatial = nn.Sequential(nn.Linear(in_planes, 1), nn.BatchNorm1d(1), nn.ReLU(inplace=True))

        self.linear_add = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                        nn.Linear(3, out_planes))

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes),
                                      nn.BatchNorm1d(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_planes, out_planes))

        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.softmax3 = nn.Softmax(dim=1)
        self.bn1 = nn.Sequential(nn.LayerNorm(out_planes), nn.ReLU(inplace=True))
        self.a = nn.Parameter(torch.zeros(size=(nsample, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(nsample, out_planes)), requires_grad=True)
        self.d = nn.Parameter(torch.zeros(size=(nsample, out_planes)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.b.data)
        nn.init.xavier_uniform_(self.d.data)
    def forward(self, x, y, y_xyz=None, idx=None) -> torch.Tensor:
        p=y_xyz.permute(0,2,1)
        y_r = y.clone().permute(0,2,1)
        if idx==None:
            _,idx = knn_point(self.k,p,p)

        b,n,_ = p.shape

        x_q, y_k, y_v = self.linear_q(x.permute(0,2,1)).reshape(b*n,-1), self.linear_k(y.permute(0,2,1)).reshape(b*n,-1), self.linear_v(y.permute(0,2,1)).reshape(b*n,-1)

        x_uni_spatial = self.linear_uni_spatial(x.reshape(-1,y_r.shape[-1])).reshape(b*n,-1)
        # device = torch.device("cuda:0")

        # y_k, grouped_xyz, new_xyz = sample_and_group_with_idx(n, idx, y_xyz, y_k.permute(0,2,1),
        #                                                       radius=self.radius,
        #                                                       grouper=self.grouper)  # b, c/r+3, n, k
        # y_v, _, _ = sample_and_group_with_idx(n, idx, y_xyz, y_v.permute(0,2,1),
        #                                       radius=self.radius, grouper=self.grouper)  # b, c, n, k

        # xyz_diff = (grouped_xyz - new_xyz.unsqueeze(-1)).permute(0,2,3,1).reshape(b*n,self.k,-1)

        key = pointops.grouping(idx.reshape(b*n,-1), y_k, p.reshape(b*n,-1), with_xyz=True)
        value = pointops.grouping(idx.reshape(b*n,-1), y_v, p.reshape(b*n,-1), with_xyz=True)
        # n, nsample, c = x_v_j.shape;
        s = self.share_planes
        pos,y_k = key[:,:,0:3], key[:,:,3:]
        # ================================================== Position Encoding ==================================================

        p_r = torch.clone(pos)
        e_dis = torch.from_numpy(np.linalg.norm(pos.detach().cpu().numpy(), axis=2)).unsqueeze(2).cuda()
        e_dis_xy = torch.from_numpy(np.linalg.norm(pos.detach().cpu().numpy()[:, :, :2], axis=2)).unsqueeze(2).cuda()
        cos_theta_e = (e_dis_xy / e_dis).nan_to_num(0)

        cos_theta_a = (pos[:, :, 1].unsqueeze(2) / e_dis_xy).nan_to_num(0)
        all_feat = torch.cat([e_dis, cos_theta_a, cos_theta_e], dim=-1)

        for i, layer in enumerate(self.linear_add):
            all_feat = layer(all_feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(
                all_feat)

        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)

        p_r += self.d * all_feat

        # ================================================== Weight ==================================================
        w_spatial_d = torch.matmul((y_k - x_q.unsqueeze(1)).transpose(0, 2).contiguous(), x_uni_spatial)
        w_spatial_d = self.softmax2(w_spatial_d)  # (n, nsample, c)
        w_spatial_e = torch.matmul(y_r, w_spatial_d.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()

        w_spatial_f = w_spatial_d.transpose(0, 2).contiguous()

        w_spatial_e = self.softmax3(w_spatial_e)

        w = torch.add(self.a * w_spatial_f, self.b * w_spatial_e)  # + p_r #(x_k - x_q.unsqueeze(1)) + p_r

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i in [0, 3] else layer(w)

        w = self.softmax(w)

        # ================================================== Combine ==================================================

        n, nsample, c = y_v.shape
        s = self.share_planes
        x = ((y_v + p_r) * w).sum(1).view(n, c)
        return x


class LAAttention(nn.Module):
    """
    Local Relation Attention
    """
    radius = 0.1
    def __init__(self, channels, shrink_ratio=4, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16}
                 ):
        super().__init__()
        # channels =64
        self.v_channels = channels
        self.qk_channels = channels // shrink_ratio
        # self.qk_channels = 16

        self.ratio = shrink_ratio
        
        self.grouper = group_args.get('NAME', 'ballquery')
        # self.grouper = group_args.get('NAME', 'knn')

        # self.grouper = knn

        self.radius = 0.1
        self.nsample = 16
        
        self.q_conv = nn.Conv1d(channels, self.qk_channels, 1)
        self.k_conv = nn.Conv1d(channels, self.qk_channels, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.geometry_encoding =GeometryEncodingMulti(self.qk_channels, rel=True, abs=False, euc=True, use_simple = False, add_polar=True, R_bins=2,A_bins=12)
        # self.geometry_encoding = GeometryEncoding_v5(self.qk_channels, rel=True, abs=False, euc=True)
        # self.geometry_encoding = GeometryEncoding(self.qk_channels, rel=True, abs=False, euc=True)

        self.neighbor_embedding = NeighborEmbedding(in_channels=self.qk_channels, emb_channels=channels, form='squ', geo=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, y_xyz=None, idx=None):
        """
        x: query vector
        y: key & value vector
        y_xyz: key & value vector's coordinates
        if x == y: self attention
        if x != y: cross attention
        """
        if y_xyz is None:
            raise ValueError("vector attention needs coordinates.")
        n = y.size(-1)
        x_q = self.q_conv(x)  # b, c/r, n
        y_k = self.k_conv(y)  # b, c/r, n
        y_v = self.v_conv(y)  # b, c, n
        if idx is None:
            y_k, grouped_xyz, new_xyz = sample_and_group(n, self.nsample, y_xyz, y_k, 
                                                         radius=self.radius, grouper=self.grouper)  # b, c/r+3, n, k
            y_v, _, _ = sample_and_group(n, self.nsample, y_xyz, y_v,
                                         radius=self.radius, grouper=self.grouper)  # b, c, n, k
        else:
            y_k, grouped_xyz, new_xyz = sample_and_group_with_idx(n, idx, y_xyz, y_k,
                                                                  radius=self.radius, grouper=self.grouper)  # b, c/r+3, n, k
            # for batch in range(grouped_xyz.size(0)):
            #     for channel in range(grouped_xyz.size(1)):
            #         for point_idx in range(grouped_xyz.size(2)):
            #             neighbors = grouped_xyz[batch, channel, point_idx, :]
            #             unique_neighbors = torch.unique(neighbors)
            #             if unique_neighbors.size(0) < neighbors.size(0):
            #                 print(f"Batch {batch}, Channel {channel}, Point {point_idx} has duplicate neighbors.")

            # 遍历每个点
            # for batch in range(tensor.size(0)):
            #     for channel in range(tensor.size(1)):
            #         for point_idx in range(tensor.size(2)):
            #             neighbors = tensor[batch, channel, point_idx, :]
            #             unique_neighbors = torch.unique(neighbors)
            #             if unique_neighbors.size(0) < neighbors.size(0):
            #                 print(f"Batch {batch}, Channel {channel}, Point {point_idx} has duplicate neighbors.")
            #

            y_v, _, _ = sample_and_group_with_idx(n, idx, y_xyz, y_v,
                                                  radius=self.radius, grouper=self.grouper)  # b, c, n, k

        h_ij = self.geometry_encoding(grouped_xyz, new_xyz)
        w = self.neighbor_embedding(y_k, x_q, h_ij)

        w = self.softmax(w) 
        x = (y_v * w).sum(-1)
        return x


class NeighborEmbedding_v2(nn.Module):
    def __init__(self, in_channels, emb_channels, form='squ', geo=True):
        """
        emb_channels: output_channels
        form: 'abs', 'squ', 'mul'
            abs: absolute difference
            squ: square difference
            mul: multiplication
        geo: geometry enconde
        """
        super().__init__()
        self.form = form
        self.geo = geo

        self.ne = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, emb_channels, 1),
        )

    def forward(self, key, query, geo=None):
        if self.form == 'abs':
            diff = key - query
            emb_features = - torch.abs(diff)
        elif self.form == 'squ':
            diff = key - query
            emb_features = - diff * diff
        elif self.form == 'mul':
            prod = key * query
            emb_features = prod

        if self.geo:
            emb_features += geo

        return self.ne(emb_features)

class LinearLocalAttention(nn.Module):
    """
    Local Relation Attention
    """
    radius = 0.1

    def __init__(self, channels, shrink_ratio=4,
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16}
                 ):
        super().__init__()
        self.v_channels = channels
        self.qk_channels = channels // shrink_ratio
        self.ratio = shrink_ratio

        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        self.q_conv = nn.Conv1d(channels, self.qk_channels, 1)
        self.k_conv = nn.Conv1d(channels*self.nsample, self.qk_channels, 1)
        self.v_conv = nn.Conv1d(channels*self.nsample, channels, 1)

        self.geometry_encoding = GeometryEncoding_v2(self.qk_channels, rel=True, abs=False, euc=True)
        self.neighbor_embedding = NeighborEmbedding(in_channels=self.qk_channels, emb_channels=channels, form='squ',
                                                    geo=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, y_xyz=None, idx=None):
        """
        x: query vector
        y: key & value vector
        y_xyz: key & value vector's coordinates
        if x == y: self attention
        if x != y: cross attention
        """

        if y_xyz is None:
            raise ValueError("vector attention needs coordinates.")
        n = y.size(-1)

        if idx is None:
            y_grouped, grouped_xyz, new_xyz = sample_and_group(n, self.nsample, y_xyz, y,
                                                         radius=self.radius, grouper=self.grouper)  # b, c/r+3, n, k
            x_grouped, _, _ = sample_and_group(n, self.nsample, y_xyz,x,
                                         radius=self.radius, grouper=self.grouper)  # b, c, n, k
        else:
            y_grouped, grouped_xyz, new_xyz = sample_and_group_with_idx(n, idx, y_xyz, y,
                                                                  radius=self.radius,
                                                                  grouper=self.grouper)  # b, c/r+3, n, k
            x_grouped, _, _ = sample_and_group_with_idx(n, idx, y_xyz, x,
                                                  radius=self.radius, grouper=self.grouper)  # b, c, n, k

            diff = y_grouped - y.unsqueeze(-1)
            b,c,n,g = diff.shape
            diff = diff.permute(0, 1, 3, 2).reshape(b, c * g, n)

            x_q = self.q_conv(x)
            y_k = self.k_conv(diff)
            y_v = self.v_conv(diff)

        h_ij = self.geometry_encoding(grouped_xyz, new_xyz)
        w = self.neighbor_embedding(y_k, x_q, h_ij)

        w = self.softmax(w)
        x = (y_v * w).sum(-1)
        return x



class PVDSA(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention #LinearLocalAttention  #

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        # if with_se:
        #     # voxel_layers.append(SE3d(channels))
        #     voxel_layers.append(GAM(channels,channels))

        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

    def forward(self, inputs):
        x, xyz, idx = inputs
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r

        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        p_p_x = self.p_self_att(x, x, xyz, idx) + x
        p_v_x2 = self.v_cross_att(p_v_x, p_p_x, xyz, idx) + p_v_x
        p_p_x2 = self.p_cross_att(p_p_x, p_v_x, xyz, idx) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVDSA2(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.act(self.bn1(self.p_self_att(x, x, xyz, idx))) + x
        p_v_x2 = self.act(self.bn2(self.v_cross_att(p_v_x, p_p_x, xyz, idx))) + p_v_x
        p_p_x2 = self.act(self.bn3(self.p_cross_att(p_p_x, p_v_x, xyz, idx))) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVDSA3(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'bias_3d': True, 'normalize': False, 
                              'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

        self.mlp1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.mlp2 = nn.Conv1d(channels, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.mlp3 = nn.Conv1d(channels, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.act(self.bn1(self.mlp1(self.p_self_att(x, x, xyz, idx)))) + x
        p_v_x2 = self.act(self.bn2(self.mlp2(self.v_cross_att(p_v_x, p_p_x, xyz, idx)))) + p_v_x
        p_p_x2 = self.act(self.bn3(self.mlp3(self.p_cross_att(p_p_x, p_v_x, xyz, idx)))) + p_p_x
        
        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x


class PVB(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        )
        self.lift_conv = nn.Sequential(
            nn.BatchNorm2d(proj_channel*2),
            nn.ReLU(True),
            nn.Conv2d(proj_channel*2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.ones([1]))

    def repeat_(self, x, part):
        repeat_dim = self.channels // part
        repeats = []
        for i in range(part-1):
            repeats.append(repeat_dim)
        repeats.append(self.channels-repeat_dim*(part-1))
        repeat_tensor = torch.tensor(repeats, dtype=torch.long, device=x.device, requires_grad=False)
        return torch.repeat_interleave(x, repeat_tensor, dim=1)

    def forward(self, inputs):
        x, xyz, idx = inputs
        n = x.size(-1)

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        if idx is None:
            x_j, xyz_j, xyz_i = sample_and_group(n, self.nsample, xyz, x, grouper=self.grouper)  # b, c/r+3, n, k
        else:
            x_j, xyz_j, xyz_i = sample_and_group_with_idx(n, idx, xyz, x)  # b, c/r+3, n, k
        dp1 = xyz_j - xyz_i.unsqueeze(-1)  # b, 3, n, k
        df = x_j - x.unsqueeze(-1) # b, c, n, k
        dp2 = self.proj_conv(df)  # b, 3, n, k
        dp = torch.cat((dp1, dp2), dim=1)   # b, 3*2, n, k
        # dp = dp1 + dp2
        w = self.repeat_(dp, 6)  # b, c, n, k
        # w = self.lift_conv(dp)
        x_j = x_j * w
        x_j = self.proj_transform(x_j)
        p_p_x = self.pool(x_j)

        if self.agg_way == 'add':
            f_x = p_v_x + p_p_x
            # f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVB2(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.shared = 6 if self.refine_way == 'cat' else 3
        mid_channel = int(np.ceil(channels / self.shared))

        # voxel depth-wise convolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        # point position-adaptive abstraction
        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.pos_conv = nn.Sequential(
            nn.Conv1d(mid_channel*self.shared, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(mid_channel*self.shared),
            nn.ReLU(inplace=True),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

        # self.beta = nn.Parameter(torch.ones([1]))        
        
    def forward(self, inputs):
        x, xyz, idx = inputs

        # 1. voxel convolution
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        # neighbor grouping
        n = x.size(-1)
        if idx is None:
            x_j, _, _ = sample_and_group(n, self.nsample, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        else:
            x_j, _, _ = sample_and_group_with_idx(n, idx, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        
        # position-adaptive weight
        dp = x_j[:, 0:3, :, :]   # b, 3, n, k
        x_j = x_j[:, 3:, :, :]   # b, c, n, k
        df = x_j - x.unsqueeze(-1)  # b, c, n, k
        weight = self.proj_conv(df)
        if self.refine_way == 'cat':
            weight = torch.cat((dp, weight), dim=1)  # b, 3*2, n, k
        else:
            weight = dp + weight
            # weight = dp 

        # 2. point abstraction
        B, _, N, K = x_j.size()
        x_j = self.pre_conv(x_j)   # b, c/s, n, k
        x_j = x_j.unsqueeze(1).repeat(1, self.shared, 1, 1, 1) * weight.unsqueeze(2)   # b, s, c/s, n, k
        x_j = x_j.view(B, -1, N, K)   # b, c, n, k 
        # fj = self.proj_transform(fj)
        x = self.pool(x_j)   # b, c, n
        p_p_x = self.pos_conv(x)

        # 3. feature fusion
        if self.agg_way == 'add':
            f_x = p_v_x + p_p_x
            # f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVB3(nn.Module):
    def __init__(self, channels, resolution=16, 
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': False,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add'}
                ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        # grouper args
        self.grouper = group_args.get('NAME', 'ballquery')
        self.radius = group_args.get('radius', 0.1)
        self.nsample = group_args.get('nsample', 16)

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        groups = aggr_args.get('groups', 0)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        proj_channel = aggr_args.get('proj_channel', 3)
        self.refine_way = aggr_args.get('refine_way', 'cat')
        self.agg_way = aggr_args.get('agg_way', 'add')

        self.shared = 6 if self.refine_way == 'cat' else 3
        mid_channel = int(np.ceil(channels / self.shared))

        # voxel depth-wise convolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, 
                      groups=groups if groups>0 else channels, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        # point position-adaptive abstraction
        self.proj_conv = nn.Sequential(
            nn.Conv2d(channels, proj_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channel, proj_channel, kernel_size=1, bias=False)
        )
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.pos_conv = nn.Sequential(
            nn.Conv1d(mid_channel*self.shared, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.proj_transform = nn.Sequential(
            nn.BatchNorm2d(mid_channel*self.shared),
            nn.ReLU(inplace=True),
        )
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        
        # self.alpha = nn.Parameter(torch.ones([1], dtype=torch.float32))       
        self.beta = nn.Parameter(torch.ones([1], dtype=torch.float32))       
        
    def forward(self, inputs):
        x, xyz, idx = inputs

        # 1. voxel convolution
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        # neighbor grouping
        n = x.size(-1)
        if idx is None:
            x_j, _, _ = sample_and_group(n, self.nsample, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        else:
            x_j, _, _ = sample_and_group_with_idx(n, idx, xyz, x, radius=self.radius, grouper=self.grouper, use_xyz=True)  # b, c/r+3, n, k
        
        # position-adaptive weight
        dp = x_j[:, 0:3, :, :]   # b, 3, n, k
        x_j = x_j[:, 3:, :, :]   # b, c, n, k
        df = x_j - x.unsqueeze(-1)  # b, c, n, k
        weight = self.proj_conv(df)
        if self.refine_way == 'cat':
            weight = torch.cat((dp, weight), dim=1)  # b, 3*2, n, k
        else:
            weight = dp + weight

        # 2. point abstraction
        B, _, N, K = x_j.size()
        x_j = self.pre_conv(x_j)   # b, c/s, n, k
        x_j = x_j.unsqueeze(1).repeat(1, self.shared, 1, 1, 1) * weight.unsqueeze(2)   # b, s, c/s, n, k
        x_j = x_j.view(B, -1, N, K)   # b, c, n, k 
        # fj = self.proj_transform(fj)
        x = self.pool(x_j)   # b, c, n
        p_p_x = self.pos_conv(x)

        # 3. feature fusion
        if self.agg_way == 'add':
            # f_x = p_v_x + p_p_x
            # f_x = self.alpha * p_v_x + self.beta * p_p_x
            f_x = self.beta * p_v_x + (1 - self.beta) * p_p_x
        else:
            f_x = torch.cat((p_v_x, p_p_x), dim=1)

        return f_x


class PVDSA_Res(nn.Module):
    """
    Point Transformer Layer of PT2(Hengshaung).
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, resolution=16, pvdsa_class=1,
                 group_args = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 # group_args={'NAME': 'knn', 'radius': 0.1, 'nsample': 16},
                 aggr_args = {'kernel_size': 3, 'groups': 0, 'bias_3d': True, 
                              'normalize': False, 'eps': 0, 'with_se': True,
                              'proj_channel': 3, 'refine_way': 'cat', 'agg_way': 'add',  
                              'res': True, 'conv_res': False},
                 **kwargs
                 ):
        super().__init__()
        self.res = aggr_args.pop('res', True)
        self.conv_res = aggr_args.pop('conv_res', False)
        agg_way = aggr_args.get('agg_way', 'add')
        self.mid_channels = out_channels // 2
        self.mlp1 = nn.Conv1d(in_channels, self.mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_channels)
        self.mlp2 = nn.Conv1d(self.mid_channels if agg_way=='add' else self.mid_channels*2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.mlp3 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        if pvdsa_class == 1:
            pvdsa_block = PVDSA 
        elif pvdsa_class == 2:
            pvdsa_block = PVDSA2 
        elif pvdsa_class == 3:
            pvdsa_block = PVDSA3 
        elif pvdsa_class == 4:
            pvdsa_block = PVB
        elif pvdsa_class == 5:
            pvdsa_block = PVB2
        elif pvdsa_class == 6:
            pvdsa_block = PVB3
        elif pvdsa_class == 7:
            pvdsa_block = PVDSA_new
        else:
            raise NotImplementedError(f'pvdsa_class {pvdsa_class} in PVDSA_Res is not implemented')
        self.pvdsa = pvdsa_block(self.mid_channels, resolution=resolution, group_args=group_args, aggr_args=aggr_args)
        self.k = 16 #16
        self.token_emb = Token_Embed(in_c=2 * in_channels, out_c=in_channels)
    def forward(self, inputs):
        x, xyz, idx = inputs   #x.shape=(7,128,10000)
        x = get_graph_feature(x, k=self.k, idx=idx) #x.shape=(7,10000,16,256)
        x = self.token_emb(x).transpose(2,1)
        p_x = self.act(self.bn1(self.mlp1(x)))  # b, c/2, n
        # f_x,cnn,trans = self.pvdsa((p_x, xyz, idx))
        f_x = self.pvdsa((p_x, xyz, idx))

        if self.res:
            if self.conv_res:
                f_x = self.act(self.bn2(self.mlp2(f_x)) + self.bn3(self.mlp3(x)))
            else:
                f_x = self.act(self.bn2(self.mlp2(f_x)) + x)
        else:
            f_x = self.act(self.bn2(self.mlp2(f_x)))

        return  (f_x, xyz, idx) #(f_x, xyz, idx,cnn,trans)#




class PVDSA_new(nn.Module):
    def __init__(self, channels, resolution=16,
                 # group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 group_args={'NAME': 'knn', 'radius': 0.1, 'nsample': 16},
                 aggr_args={'kernel_size': 3, 'bias_3d': True, 'normalize': False,
                            'eps': 0, 'with_se': True, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)

        # self.dilation = [1, 2, 3, 4]
        # self.kernel_size = [1,3, 5, 7]
        # self.conv_groups = [1, 4, 8,16]
        self.conv_groups = [1, 2, 4, 8]
        self.kernel_size = 3

        self.agg_way = aggr_args.get('agg_way', 'add')
        attention = LAAttention  # LinearLocalAttention  #
        # attention = WeakLayer
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        # self.voxelization = Voxelization_2(resolution, normalize=normalize, eps=eps)
        # voxel_layers = [
        #     nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, bias=bias_3d),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, bias=bias_3d),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        # ]
        # voxel_layers = [
        #     DaliCMoudle(channels, channels, kernel_size,dilation=self.dilation, stride=1, conv_groups=self.conv_groups,bias=bias_3d),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     DaliCMoudle(channels, channels, kernel_size,dilation=self.dilation, stride=1, conv_groups=self.conv_groups,bias=bias_3d),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True)
        # ]
        voxel_layers = [
            DaliCMoudle_conv_4(channels, channels,kernel_size=self.kernel_size,bias=True,conv_groups=self.conv_groups),
            # nn.BatchNorm3d(channels, eps=1e-4),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.1, True),
            DaliCMoudle_conv_4(channels, channels, kernel_size=self.kernel_size, bias=True, conv_groups=self.conv_groups,last=False),
            # nn.BatchNorm3d(channels, eps=1e-4),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.1, True),
        ]
        # if with_se:
        #     voxel_layers.append(SE3d(channels))
        # voxel_layers = [
        #     DaliCMoudle_3(channels, channels, kernel_size=[3, 5, 7], stride=1, conv_groups=[1, 1, 1]),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True),
        #     DaliCMoudle_3(channels, channels, kernel_size=[3, 5, 7], stride=1, conv_groups=[1, 1, 1]),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True)
        # ]
        # if with_se:   #添加一个通道注意力，训练时常见的一个小trick，由于篇幅的原因文章里没有提到
        #     voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)

        # self.conv3ds = SequentialPassKw(*voxel_layers)

        # self.convs3d_pre = nn.Sequential(nn.Conv3d(channels, channels, 3, stride=1, padding=1, bias=bias_3d),
        #                     nn.BatchNorm3d(channels, eps=1e-4),
        #                     nn.LeakyReLU(0.1, True))
        # self.convs1= nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1,stride=1, bias=False),
        #     nn.BatchNorm3d(channels, eps=1e-4),
        #     nn.LeakyReLU(0.1, True))
        # self.convs2 = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
        #                             nn.BatchNorm3d(channels, eps=1e-4),
        #                             nn.LeakyReLU(0.1, True))
        # self.gate = nn.Sequential(
        #     nn.Conv1d(2 * channels, channels, kernel_size=1, bias=True),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(channels, channels, 1)
        # )
        # for m in self.gate.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)
        # self.d = nn.Parameter(torch.randn(size=(1, channels, 1)), requires_grad=True)
        # self.p = nn.Parameter(torch.randn(size=(1, channels, 1)), requires_grad=True)
        # nn.init.xavier_uniform_(self.d.data)
        # nn.init.xavier_uniform_(self.p.data)
        # self.norm_v = nn.BatchNorm1d(channels)
        # self.norm_p = nn.BatchNorm1d(channels)
        self.CFTB = CFTB_new(channels)


    def forward(self, inputs):
        x, xyz, idx = inputs
        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n

        p_p_x = self.p_self_att(x, x, xyz, idx) + x
        # f_x = p_v_x + p_p_x
        p_v_x2 = self.v_cross_att(p_v_x, p_p_x, xyz, idx) + p_v_x
        p_p_x2 = self.p_cross_att(p_p_x, p_v_x, xyz, idx) + p_p_x


        # f_x = self.d * p_v_x2 + self.p * p_p_x2

        # if self.agg_way == 'add':
        #     f_x = p_v_x2 + p_p_x2
        # else:
        #     f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        f_x = self.CFTB(p_v_x2, p_p_x2)


        return f_x #f_x,p_v_x2,p_p_x2  #,p_v_x2,p_p_x2



class BasicBlock1D(nn.Module):
    """BRB 点域版"""
    def __init__(self, channels, dropout=0.1, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3,padding=1, bias=True),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3,padding=1, bias=True)
        )
    def forward(self, x):
        return x + self.block(x)



class CFTB(nn.Module):
    def __init__(self, channels, num_groups=8, dropout=0.0):
        super().__init__()
        self.gn_c = nn.GroupNorm(num_groups, channels)
        self.gn_f = nn.GroupNorm(num_groups, channels)
        self.tau = 1.5
        self.gate = nn.Sequential(
            nn.Conv1d(channels*2, channels, kernel_size=1, bias=True),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels , 1)
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.t = nn.Parameter(torch.tensor(float(self.tau)), requires_grad=True)

        self.fusion=nn.Sequential(
            nn.Conv1d(channels*3, channels, kernel_size=1, bias=True),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1)
        )
        self.norm=nn.GroupNorm(8, channels)

    def forward(self, coarse, fine):
        p_v,p_p = coarse,fine
        B, C, N = coarse.shape
        c = self.gn_c(coarse)
        f = self.gn_f(fine)
        g_x = torch.cat((c, f), dim=1)

        logits = self.gate(g_x)

        w = torch.sigmoid(logits / self.t.clamp_min(1.0))
        f_x_f = c * w + f * (1-w)

        f_x = self.fusion(torch.cat([f_x_f,c, f], dim=1))

        return f_x

class CFTB_new(nn.Module):
    def __init__(self, channels, num_groups=8, dropout=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.t = nn.Parameter(torch.tensor(float(1.5)), requires_grad=True)
        self.fc = nn.Conv1d(channels, 2, kernel_size=1)
        self.fc1 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        )
        self.delta = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        )
        # self.fusion=nn.Sequential(
        #     nn.Conv1d(channels*2, channels, kernel_size=1, bias=True),
        #     nn.GroupNorm(8, channels),
        #     nn.GELU(),
        #     nn.Conv1d(channels, channels, 1)
        # )
        # self.norm=nn.GroupNorm(8, channels)
        # self.align = nn.Sequential(
        #     nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        #     nn.GroupNorm(8, channels)
        # )

    def forward(self, coarse, fine):
        base = coarse+fine
        B, C, N = coarse.shape
        c = self.fc1(coarse)
        f = self.fc2(fine)
        g_x = c +f
        logits = self.fc(torch.tanh(self.gate(g_x)))
        t = torch.clamp(F.softplus(self.t),min=0.7)
        w = torch.sigmoid(
            logits.view(B,2,1, N) / t)
        w =w/(w.sum(dim=1,keepdim=True)+1e-6)
        f_x = coarse* w[:, 0]+ fine* w[:, 1]

        return self.delta(f_x)+base


class CFTB_2(nn.Module):
    """
    输入：
      coarse: [B,C,N]（如 CNN 分支输出）
      fine:   [B,C,N]（如 Transformer 分支输出）
      idx:    [B,N,k] KNN/ball query 邻域
    输出：
      out_c, out_f: 两路增强后的特征，均为 [B,C,N]
    """

    def __init__(self, channels, num_groups=8, dropout=0.0):
        super().__init__()
        assert channels % 2 == 0, "channels 必须为偶数，用于二分 m->(m1,m2)"
        # self.C = channels
        # self.d = channels // 2

        self.gn_c = nn.GroupNorm(num_groups, channels)
        self.gn_f = nn.GroupNorm(num_groups, channels)
        self.tau = 1.5
        self.brb = BasicBlock1D(channels, dropout=dropout, num_groups=num_groups)
        self.gate = nn.Sequential(
            nn.Conv1d(channels*2, channels//2, kernel_size=1, bias=True),
            # nn.BatchNorm1d(channels*2),
            nn.GroupNorm(8, channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels //2, channels*2 , 1)
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.t = nn.Parameter(torch.tensor(float(self.tau)), requires_grad=True)
        self.fc1 = nn.Conv1d(channels, 2*channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(2*channels, 2*channels, kernel_size=1, bias=True)

        # self.fc = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

        # 对增强后的两路做线性投影（可选）
        # self.out_c = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        # self.out_f = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

        # self.norm_v = nn.BatchNorm1d(channels)
        # self.norm_p = nn.BatchNorm1d(channels)

    def forward(self, coarse, fine):
        p_v,p_p = coarse,fine
        B, C, N = coarse.shape
        # 1) 归一化对齐
        c = self.gn_c(coarse)  # [B,C,N]
        f = self.gn_f(fine)  # [B,C,N]
        # 2) 形成联合表征并提炼
        m = self.brb(c + f)  # [B,C,N]
        m = self.fc1(m)
        logits = self.fc2(torch.tanh(m))
        w = F.softmax(logits.view(B, 2, C, N) / self.t, dim=1)  # [B,2,C,N]
        fx= w[:, 0] * c + w[:, 1] * f
        # att = torch.sigmoid(/self.t)
        # a,b = torch.chunk(att, chunks=2, dim=1)
        #
        # fx = a*coarse+b*fine

        # pv = self.norm_v(p_v_x2)  # [B, C, N]
        # pp = self.norm_p(p_p_x2)  # [B, C, N]
        # g_x = torch.cat((pv, pp), dim=1)
        # w = self.gate(g_x)

        # g = torch.sigmoid(w)
        # a,b = torch.chunk(att, chunks=2, dim=1)

        # f_x = pv * g + (1-g)*pp

        return fx

class CFTB2_DWGate(nn.Module):
    def __init__(self, channels, num_groups=8, dropout=0.0, init_tau=1.5, use_gn=True):
        super().__init__()
        C = channels
        self.norm_c = nn.GroupNorm(num_groups, C) if use_gn else nn.BatchNorm1d(C)
        self.norm_f = nn.GroupNorm(num_groups, C) if use_gn else nn.BatchNorm1d(C)
        self.brb = BasicBlock1D(C, dropout=dropout, num_groups=num_groups)
        self.pre = nn.GroupNorm(8, 2*C) if use_gn else nn.BatchNorm1d(2*C)
        self.proj1 = nn.Conv1d(2*C, 2*C, 1, bias=True)
        self.act1  = nn.GELU()
        self.dw    = nn.Conv1d(2*C, 2*C, kernel_size=3, padding=1, groups=2*C, bias=False)
        self.act2  = nn.GELU()
        self.proj2 = nn.Conv1d(2*C, 2*C, 1, bias=True)
        self.tau   = nn.Parameter(torch.tensor(float(init_tau)), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        for m in [self.proj1, self.proj2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, coarse, fine):
        B, C, N = coarse.shape
        c = self.norm_c(coarse)
        f = self.norm_f(fine)
        # _ = self.brb(c + f)  # 预融合

        x = torch.cat([c, f], dim=1)
        x = self.pre(x)
        x = self.act1(self.proj1(x))
        x = self.act2(self.dw(x))
        logits = self.proj2(x)              # [B,2C,N]
        w = F.softmax(logits.view(B, 2, C, N) / self.tau.clamp_min(0.25), dim=1)

        fx_dyn = w[:,0]*c + w[:,1]*f
        fx = fx_dyn + self.alpha.sigmoid() * (c + f) #* 0.5
        return fx




class FusionModule(nn.Module):
    """
    输入：两路增强后的特征 [B,C,N], [B,C,N]
    输出：融合后特征 [B,C,N]
    """

    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, out_c, out_f):
        fused = torch.cat([out_c, out_f], dim=1)  # [B,2C,N]
        return self.fusion_conv(fused)  # [B,C,N]


import inspect

class SequentialPassKw(nn.Sequential):
    def forward(self, x, **kwargs):
        for m in self:
            # 仅把该模块 forward 能接收的 kw 传进去（其余丢弃，避免报错）
            try:
                params = inspect.signature(m.forward).parameters
                accept = set(params.keys())
            except (ValueError, TypeError):
                accept = set()
            use_kwargs = {k: v for k, v in kwargs.items() if k in accept}
            x = m(x, **use_kwargs) if use_kwargs else m(x)
        return x

class DaliCMoudle(nn.Module):
    def __init__(self, inchanell, outchannell, kernel_size=[3, 5, 7, 9],stride=1, conv_groups=[1, 4, 8, 16],bias=False):
        super(DaliCMoudle, self).__init__()
        self.conv_1 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[0], padding=kernel_size[0] // 2,
                                stride=stride, groups=conv_groups[0], bias=bias)
        self.conv_2 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[1], padding=kernel_size[1] // 2,
                                stride=stride, groups=conv_groups[1], bias=bias)
        self.conv_3 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[2], padding=kernel_size[2] // 2,
                                stride=stride, groups=conv_groups[2], bias=bias)
        self.conv_4 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[3], padding=kernel_size[3] // 2,
                                stride=stride, groups=conv_groups[3], bias=bias)
        self.se = SEWeightModule3D(outchannell//4)
        # self.se = CBAM3D(outchannell//4)
        self.split_channel = outchannell//4
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        self.short_cut = nn.Conv3d(outchannell, outchannell, kernel_size=1, stride=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv3d(outchannell, outchannell, kernel_size=1),
            nn.BatchNorm3d(outchannell),
            nn.ReLU(),
            nn.Conv3d(outchannell, outchannell, kernel_size=1),
            nn.BatchNorm3d(outchannell),
            nn.ReLU(),
            nn.Conv3d(inchanell, outchannell, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # short_cut = self.short_cut(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        # feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats_ori = torch.cat((x1, x2, x3, x4), dim=1)

        feats = feats_ori.view(batch_size, 4, self.split_channel, feats_ori.shape[2], feats_ori.shape[3],feats_ori.shape[4])
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # feat_all = feats+x_se
        # x_se = self.fuse(feat_all)
        # out = short_cut+x_se
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1,1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        # out = torch.sum(feats_weight, dim=1)

        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :,:]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        out = feats_ori+out
        out =self.fuse(out)
        return out




# class DaliCMoudle_conv(nn.Module):
#     def __init__(self, inchanell, outchannell, kernel_size=[3, 5, 7, 9],stride=1, conv_groups=[1, 4, 8, 16],bias=False):
#         super(DaliCMoudle_conv, self).__init__()
#         self.conv_1 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[0], padding=kernel_size[0]//2,
#                             stride=stride, groups=conv_groups[0],bias=bias)
#         self.conv_2 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[1], padding=kernel_size[1]//2,
#                             stride=stride, groups=conv_groups[1],bias=bias)
#         self.conv_3 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[2], padding=kernel_size[2]//2,
#                             stride=stride, groups=conv_groups[2],bias=bias)
#         # self.conv_4 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[3], padding=kernel_size[3]//2,
#         #                     stride=stride, groups=conv_groups[3],bias=bias)
#         # self.maxpool = nn.Sequential(nn.MaxPool3d(kernel_size=3,stride=stride),
#         #                              nn.Conv3d(inchanell, outchannell // 4, kernel_size=1,
#         #                                        stride=stride, bias=bias)
#         #                              )
#         self.conv_4 = nn.Conv3d(inchanell, outchannell // 4, kernel_size=1,stride=stride, bias=bias)
#
#         # self.se = SEWeightModule3D(outchannell // 4)
#         # self.split_channel = outchannell // 4
#         # self.softmax = nn.Softmax(dim=1)
#         self.fuse = nn.Sequential(
#             nn.Conv3d(outchannell, outchannell, kernel_size=1),
#             nn.BatchNorm3d(outchannell),
#             nn.LeakyReLU(True),
#             nn.Conv3d(outchannell, outchannell, 1),
#         #     nn.BatchNorm3d(outchannell),
#         #     nn.LeakyReLU(True),
#         #     nn.Conv3d(outchannell, outchannell, 1)
#         )
#     def forward(self, x):
#         b,c,h,w,d = x.shape
#         x1 = self.conv_1(x)
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x)
#         # x4 = self.maxpool(x)
#         # x4 = F.interpolate(self.conv3d(F.max_pool3d(x, (1, 1, 1))),
#         #                             size=x.size()[2:], mode='trilinear', align_corners=False)
#         x4 = F.interpolate(
#             self.conv_4(F.max_pool3d(x, kernel_size=(h, w, d))),
#             size=(h, w, d),
#             mode='trilinear',
#             align_corners=False
#         )
#         feats = torch.cat((x1, x2, x3, x4), dim=1)
#
#         out = self.fuse(feats)
#         # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3],feats.shape[4])
#         #
#         # x1_se = self.se(x1)
#         # x2_se = self.se(x2)
#         # x3_se = self.se(x3)
#         # x4_se = self.se(x4)
#         #
#         # x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
#         # attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1,1)
#         # attention_vectors = self.softmax(attention_vectors)
#         # feats_weight = feats * attention_vectors
#         # for i in range(4):
#         #     x_se_weight_fp = feats_weight[:, i, :, :,:]
#         #     if i == 0:
#         #         out = x_se_weight_fp
#         #     else:
#         #         out = torch.cat((x_se_weight_fp, out), 1)
#
#         return out





def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                               res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                      normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                                   res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    # in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = nn.ModuleList()
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                          normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
            for _ in range(num_blocks):
                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
            # extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=in_channels, out_channels=out_channels,
                               include_coordinates=True))
        # in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        in_channels = out_channels[-1]
        sa_layers.append(sa_blocks)

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, nsample=16, with_se=False, normalize=True, eps=0, agg_way='add',
                                res=True, conv_res=True, width_multiplier=1, voxel_resolution_multiplier=1, pvdsa_class=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = nn.ModuleList()
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx], out_channels=out_channels)
        )
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVDSA_Res, nsample=nsample, resolution=int(vr * voxel_resolution), with_se=with_se, 
                                          normalize=normalize, eps=eps, agg_way=agg_way, res=res, conv_res=conv_res, pvdsa_class=pvdsa_class)
            for _ in range(num_blocks):
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        fp_layers.append(fp_blocks)

    return fp_layers, in_channels


class PCFormer(nn.Module):
    def __init__(self, channels, resolution=16,
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 aggr_args={'kernel_size': 3, 'bias_3d': True, 'normalize': False,
                            'eps': 0, 'with_se': False, 'agg_way': 'add'}
                 ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution

        # aggregation args
        kernel_size = aggr_args.get('kernel_size', 3)
        bias_3d = aggr_args.get('bias_3d', False)
        normalize = aggr_args.get('normalize', False)
        eps = aggr_args.get('eps', 0)
        with_se = aggr_args.get('with_se', False)
        self.agg_way = aggr_args.get('agg_way', 'add')

        attention = LAAttention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, bias=bias_3d),
            nn.BatchNorm3d(channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.conv3ds = nn.Sequential(*voxel_layers)
        self.v_cross_att = attention(channels, group_args=group_args)
        self.p_self_att = attention(channels, group_args=group_args)
        self.p_cross_att = attention(channels, group_args=group_args)

    def forward(self, inputs):
        x, xyz, idx = inputs

        v_x, v_xyz = self.voxelization(x, xyz)
        v_x = self.conv3ds(v_x)  # b, c/2, r, r, r
        p_v_x = MF.trilinear_devoxelize(v_x, v_xyz, self.resolution, self.training)  # b, c/2, n
        p_p_x = self.p_self_att(x, x, xyz, idx) + x
        p_v_x2 = self.v_cross_att(p_v_x, p_p_x, xyz, idx) + p_v_x
        p_p_x2 = self.p_cross_att(p_p_x, p_v_x, xyz, idx) + p_p_x

        if self.agg_way == 'add':
            f_x = p_v_x2 + p_p_x2
        else:
            f_x = torch.cat((p_v_x2, p_p_x2), dim=1)

        return f_x

class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mid_c =in_c//2
        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(in_c, in_c, 1),
        #     nn.BatchNorm1d(in_c),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_c, in_c, 1)
        # )
        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(in_c * 2, out_c, 1),
        #     nn.BatchNorm1d(out_c),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(out_c, out_c, 1)
        # )
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.mid_c, self.mid_c, 1),
            nn.BatchNorm1d(self.mid_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_c, self.mid_c, 1)
            )
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.mid_c * 2, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 1)
            )
        self.down = nn.Sequential(
            nn.Conv1d(in_c, self.mid_c, 1),
            nn.BatchNorm1d(self.mid_c),
            nn.ReLU(inplace=True),
        #     # nn.Conv1d(self.mid_c, self.mid_c, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.down(point_groups.transpose(2, 1))
        feature = self.first_conv(feature)  # BG 256 n
        # feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.out_c)
