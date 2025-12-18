import torch
import torch.nn as nn
import torch.nn.functional as F

class DaliCMoudle_3(nn.Module):
    def __init__(self, inchanell, outchannell, kernel_size=[3, 5, 7, 9],stride=1, conv_groups=[1, 4, 8, 16],bias=True):
        super(DaliCMoudle_3, self).__init__()
        self.mid_channell = outchannell//2
        self.down_conv = nn.Sequential(
                        nn.Conv3d(inchanell, self.mid_channell, kernel_size=3,padding=1),
                        nn.BatchNorm3d(self.mid_channell),
                        nn.ReLU(True))

        # self.conv_1 = nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[0], padding=kernel_size[0]//2,
        #                     stride=stride, groups=conv_groups[0],bias=bias)
        # self.conv_2 = nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[1], padding=kernel_size[1]//2,
        #                     stride=stride, groups=conv_groups[1],bias=bias)
        # self.conv_3 = nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[2], padding=kernel_size[2]//2,
        #                     stride=stride, groups=conv_groups[2],bias=bias)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[0], padding=kernel_size[0] // 2,
                      stride=stride, groups=conv_groups[0],bias=bias),
            nn.BatchNorm3d(self.mid_channell),
            nn.ReLU(True))
        self.conv_2 = nn.Sequential(
            nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[1], padding=kernel_size[1] // 2,
                      stride=stride, groups=conv_groups[1], bias=bias),
            nn.BatchNorm3d(self.mid_channell),
            nn.ReLU(True))
        self.conv_3 = nn.Sequential(
            nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=kernel_size[2], padding=kernel_size[2] // 2,
                      stride=stride, groups=conv_groups[2], bias=bias),
            nn.BatchNorm3d(self.mid_channell),
            nn.ReLU(True))
        self.conv_4 = nn.Sequential(
            nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=1),
            nn.BatchNorm3d(self.mid_channell),
            nn.ReLU(True))

        # self.conv_4 = nn.Conv3d(inchanell, outchannell//4, kernel_size=kernel_size[3], padding=kernel_size[3]//2,
        #                     stride=stride, groups=conv_groups[3],bias=bias)
        # self.maxpool = nn.Sequential(nn.MaxPool3d(kernel_size=3,stride=stride),
        #                              nn.Conv3d(inchanell, outchannell // 4, kernel_size=1,
        #                                        stride=stride, bias=bias)
        #                              )
        # self.conv_4 = nn.Conv3d(self.mid_channell, self.mid_channell, kernel_size=1,stride=stride, bias=False)
        # self.pool = nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
        # self.se = SEWeightModule3D(outchannell // 4)
        # self.split_channel = outchannell // 4
        # self.softmax = nn.Softmax(dim=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(outchannell*2, outchannell, 1),
            nn.BatchNorm3d(outchannell),
            nn.ReLU(True),
            nn.Conv3d(outchannell, outchannell, 1),
        #     nn.BatchNorm3d(outchannell),
        #     nn.LeakyReLU(True),
        #     nn.Conv3d(outchannell, outchannell, 1)
        )
    def forward(self, x):
        entity =x
        x=self.down_conv(x)
        b,c,h,w,d = x.shape
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        # x4 = self.maxpool(x)
        # x4 = F.interpolate(self.conv3d(F.max_pool3d(x, (1, 1, 1))),
        #                             size=x.size()[2:], mode='trilinear', align_corners=False)
        # x4 = self.conv_4(self.pool(x))
        x4 = F.interpolate(
            self.conv_4(F.adaptive_avg_pool3d(x, 1)),
            size=(h, w, d),
            mode='trilinear',
            align_corners=False
        )
        feats = torch.cat((x1, x2, x3, x4), dim=1)

        out = self.fuse(feats)+entity
        # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3],feats.shape[4])
        #
        # x1_se = self.se(x1)
        # x2_se = self.se(x2)
        # x3_se = self.se(x3)
        # x4_se = self.se(x4)
        #
        # x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1,1)
        # attention_vectors = self.softmax(attention_vectors)
        # feats_weight = feats * attention_vectors
        # for i in range(4):
        #     x_se_weight_fp = feats_weight[:, i, :, :,:]
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)

        return out


class DaliCMoudle_conv(nn.Module):
    def __init__(self, inchanell, outchannell, kernel_size=3,stride=1, conv_groups=[1, 2, 4, 8],bias=False):
        super(DaliCMoudle_conv, self).__init__()
        self.conv_1 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[0],bias=bias)
        self.conv_2 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[1],bias=bias)
        self.conv_3 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[2],bias=bias)
        self.conv_4 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[3],bias=bias)
        self.fuse = nn.Sequential(
            nn.Conv3d(outchannell*4, outchannell, kernel_size=1,bias=True),
            nn.BatchNorm3d(outchannell),
            nn.LeakyReLU(True),
            # nn.Conv3d(outchannell, outchannell, 1),
            # nn.BatchNorm3d(outchannell),
            # nn.LeakyReLU(True),
            nn.Conv3d(outchannell, outchannell, 1,bias=True)
        )
        # self.se = SE3d_AvgMax(outchannell,reduction=4)
        # self.se = SE3d(outchannell,reduction=4)

        # self.se =SE3d_masked(outchannell,reduction=4)
    def forward(self, x,mask=None):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)

        # out = self.fuse(feats)
        # out_se = self.se(out)
        # out_final = out_se+x
        # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3],feats.shape[4])
        #
        # x1_se = self.se(x1)
        # x2_se = self.se(x2)
        # x3_se = self.se(x3)
        # x4_se = self.se(x4)
        #
        # x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1,1)
        # attention_vectors = self.softmax(attention_vectors)
        # feats_weight = feats * attention_vectors
        # for i in range(4):
        #     x_se_weight_fp = feats_weight[:, i, :, :,:]
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)
        # feats = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.fuse(feats)  # [B, C, D, H, W]
        # out = out + x  # 先做短残差
        # out = self.se(out)
        # 温和 SE：scaled ∈ [1, 1.5]
        # se_out = self.se(out)
        # out = out +se_out
        return out
        # return out_final


class DaliCMoudle_conv_4(nn.Module):
    def __init__(self, inchanell, outchannell, kernel_size=3,stride=1, conv_groups=[1, 2, 4, 8],bias=False,last=False):
        super(DaliCMoudle_conv_4, self).__init__()
        self.conv_1 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[0],bias=bias)
        self.conv_2 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[1],bias=bias)
        self.conv_3 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[2],bias=bias)
        self.conv_4 = nn.Conv3d(inchanell, outchannell, kernel_size=kernel_size, padding=kernel_size//2,
                            stride=stride, groups=conv_groups[3],bias=bias)
        self.fuse = nn.Sequential(
            nn.Conv3d(outchannell*4, outchannell, kernel_size=1,bias=True),
            # nn.BatchNorm3d(outchannell),
            nn.GroupNorm(8, outchannell),
            nn.LeakyReLU(True),
            # nn.Conv3d(outchannell, outchannell, 1),
            # nn.BatchNorm3d(outchannell),
            # nn.LeakyReLU(True),
            nn.Conv3d(outchannell, outchannell, 1,bias=True)
        )
        self.last = last
        if self.last:
            self.sc = MKCB3D(outchannell)
            # self.sb = MKSB3D(outchannell)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        c_feats = torch.cat((x1, x2, x3, x4), dim=1)
        # a_feats = x1+x2+x3+x4
        out = self.fuse(c_feats)  # [B, C, D, H, W]
        if self.last:
            out = self.sc(c_feats,out)
            # out = self.sb(a_feats,out)
        return out


class MKCB3D(nn.Module):
    def __init__(self, channels, pool_ks=3, use_gn=False, gn_groups=16):
        super().__init__()
        C = channels
        pad = pool_ks // 2
        self.avg = nn.AvgPool3d(kernel_size=pool_ks, stride=1, padding=pad)
        self.max = nn.MaxPool3d(kernel_size=pool_ks, stride=1, padding=pad)
        self.mix3 = nn.Conv3d(4*C, C, kernel_size=3, padding=1, bias=False)
        self.mix1 = nn.Conv3d(C, C, kernel_size=1, bias=True)
        self.act  = nn.GELU()
        self.norm = (nn.GroupNorm(gn_groups, C) if use_gn else nn.BatchNorm3d(C))

    def forward(self, f_cat, f_in):
        f_pool = self.avg(f_cat) + self.max(f_cat)
        w = self.mix1(self.act(self.mix3(f_pool)))
        w = torch.sigmoid(self.norm(w))
        return w * f_in                                  # [B,C,D,H,W]


class MKSB3D(nn.Module):
    def __init__(self, channels, num_groups=8, dropout=0.0, temperature=1.0):
        super().__init__()
        # self.gn   = nn.GroupNorm(num_groups, channels)
        self.gn1   = nn.GroupNorm(num_groups, channels)
        # self.gn2   = nn.GroupNorm(num_groups, channels)

        self.brb  = BasicBlock3D(channels, dropout=dropout, num_groups=num_groups)
        self.tau  = temperature

    def forward(self,f_add,x):                     # x: [B,C,D,H,W]
        x = self.gn1(x)
        # f_add = self.gn2(f_add)
        z = self.brb(f_add)
        B, C, D, H, W = z.shape
        att = z.view(B, C, -1) / self.tau
        att = F.softmax(att, dim=-1)
        att = att.view(B, C, D, H, W)
        out = x * att
        return out

class BasicBlock3D(nn.Module):
    """BRB 点域版"""
    def __init__(self, channels, dropout=0.1, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1,bias=True),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1,bias=True)
        )
    def forward(self, x):
        return x+self.block(x)


class SE3d(nn.Module):
    """Squeeze-and-Excitation for 3D feature maps: [B, C, D, H, W]."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))     # [B, C, 1, 1, 1]
        return x * w                 # 通道重标定

class SE3d_AvgMax(nn.Module):
    """Squeeze-and-Excitation 3D with Avg+Max pooling in parallel.
       输入/输出与旧 SE3d 完全一致：forward(x) -> x * w
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.max = nn.AdaptiveMaxPool3d(1)
        # 并联后通道是 2C → 压到 C
        self.fc = nn.Sequential(
            nn.Conv3d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg(x)                     # [B,C,1,1,1]
        mx  = self.max(x)                     # [B,C,1,1,1]
        s   = torch.cat([avg, mx], dim=1)     # [B,2C,1,1,1]
        w   = self.fc(s)                      # [B,C,1,1,1] in (0,1)
        return x * w                          # 与原 SE3d 行为一致


import torch
import torch.nn as nn
import torch.nn.functional as F

class SE3d_masked(nn.Module):
    """
    带 mask 的 3D Squeeze-and-Excitation
    - 与常规 SE3d 接口兼容：SE3d_masked(channels, reduction=4)
    - forward(x, mask=None):
        x:    [B, C, D, H, W]
        mask: [B, 1, D, H, W] 或 [B, D, H, W]，非空=1、空=0；可 None（退化为 GAP）
    - 返回: scale ∈ [0,1], 形状 [B, C, 1, 1, 1]
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 无 mask 时使用
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出 [0,1]
        )
        # 初始化（可选）：小范围权重、零偏置更稳
        for m in self.fc.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _check_mask(self, x, mask):
        # 统一成 [B,1,D,H,W] 的 float mask
        if mask is None:
            return None
        if mask.dim() == 4:  # [B,D,H,W]
            mask = mask.unsqueeze(1)
        # 对齐 dtype/device，并将布尔转 float
        if mask.dtype != x.dtype:
            mask = mask.to(dtype=x.dtype)
        if mask.device != x.device:
            mask = mask.to(device=x.device)
        return mask

    def _masked_avg(self, x, mask, eps=1e-6):
        """
        x:    [B, C, D, H, W]
        mask: [B, 1, D, H, W]  (0/1)
        return: [B, C, 1, 1, 1]
        """
        # 广播到通道维
        m = mask.expand_as(x)                      # [B, C, D, H, W]
        num = (x * m).sum(dim=(2, 3, 4), keepdim=True)
        den = m.sum(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
        return num / den

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        返回通道缩放系数 scale ∈ [0,1], 形状 [B, C, 1, 1, 1]
        用法（与你现有代码兼容）：
            scale = self.se(x, mask)     # 或 self.se(x)
            out   = x * (1.0 + 0.5*scale)
        """
        mask = self._check_mask(x, mask)

        if mask is None:
            # 标准 GAP
            y = self.avg_pool(x)  # [B,C,1,1,1]
        else:
            # 仅对非空体素池化
            y = self._masked_avg(x, mask)  # [B,C,1,1,1]

        scale = self.fc(y)  # [B,C,1,1,1], in [0,1]
        return scale