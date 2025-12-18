import torch.nn as nn
import torch
__all__ = ['SE3d']


class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)




class GAM(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)

        # self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.linear1 = nn.Conv1d(in_channels, inchannel_rate, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(inchannel_rate, in_channels)
        self.linear2 = nn.Conv1d(inchannel_rate, in_channels, 1, bias=False)
        self.conv1 = nn.Conv3d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv3d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')
        self.norm1 = nn.BatchNorm3d(inchannel_rate)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)

        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))

        out = x * x_spatial_att

        return out



class SEWeightModule3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEWeightModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)  # 对 D, H, W 进行全局平均池化
        out = self.fc1(out)  # 通道压缩
        out = self.relu(out)  # 激活
        out = self.fc2(out)  # 通道恢复
        weight = self.sigmoid(out)  # 通道权重归一化
        return weight


import torch
import torch.nn as nn

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(x))
        return out

class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_att = ChannelAttention3D(in_planes, ratio)
        self.spatial_att = SpatialAttention3D(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out