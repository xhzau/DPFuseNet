import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pvdst_utils_3 import PVDSA_Res
# from pvdst_utils_3 import PVDSA_Res

from modules.ops import knn_point
from PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet

import os
import numpy as np

def save_branch_features(
    feat_cnn: torch.Tensor,
    feat_trans: torch.Tensor,
    feat_fuse: torch.Tensor = None,
    root_dir: str = "./feature_dump",
    encoder_layer: int = 0,  # Encoder layer number for unique naming
):
    """
    Save CNN, Transformer, and fused features to disk for visualization purposes.
    File paths and names are automatically generated based on encoder layer number and branch type.
    """
    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)

    def _to_np(x):
        """Convert tensor to numpy."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return x

    feat_cnn_np   = _to_np(feat_cnn)
    feat_trans_np = _to_np(feat_trans)
    feat_fuse_np  = _to_np(feat_fuse)

    # Define file paths based on layer and branch type
    def get_file_path(branch_name):
        return os.path.join(root_dir, f"encoder_layer{encoder_layer}-{branch_name}.npz")

    # Save CNN branch feature
    if feat_cnn_np is not None:
        cnn_file_path = get_file_path("cnn")
        np.savez_compressed(cnn_file_path, feat_cnn=feat_cnn_np)
        print(f"[save_branch_features] CNN feature saved to: {cnn_file_path}")

    # Save Transformer branch feature
    if feat_trans_np is not None:
        trans_file_path = get_file_path("transformer")
        np.savez_compressed(trans_file_path, feat_trans=feat_trans_np)
        print(f"[save_branch_features] Transformer feature saved to: {trans_file_path}")

    # Save Fused feature
    if feat_fuse_np is not None:
        fuse_file_path = get_file_path("fuse")
        np.savez_compressed(fuse_file_path, feat_fuse=feat_fuse_np)
        print(f"[save_branch_features] Fused feature saved to: {fuse_file_path}")



class PatchEmbed(nn.Module):
    """ Patch Embedding, dimension tranformation
    """
    def __init__(self, in_chans=3, embed_dim=64):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.proj(x)
        return x


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim



class PVDST_partseg(nn.Module):
    """
    Point-Voxel Dual Stream Transformer for part segmentation.
    """
    shape_label_channels = 64

    resolution = [32, 16, 16] #[32, 16, 16]
    out_channels = [128, 128, 128]  #[128, 128, 128]
    nsample = 16 #16
    customized_parameters = {'kernel_size': 3, 'groups': 0, 'bias_3d': True,
                             'normalize': False, 'eps': 0, 'with_se': False,
                             'agg_way': 'add', 'res': True, 'conv_res': False,
                             'proj_channel': 3, 'refine_way': 'cat', 'grouper': 'knn'
                             }

    def __init__(self, args, extra_feature_channels=0,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = args.num_shapes
        cps = self.customized_parameters
        self.k=16
        resolution = [r * voxel_resolution_multiplier for r in self.resolution]
        out_channels = [oc * width_multiplier for oc in self.out_channels]
        embed_channels = out_channels[0]
        self.nblock = len(out_channels)

        self.input_embedding = nn.Sequential(
            nn.Conv1d(self.in_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_channels),
            nn.ReLU(),
            nn.Conv1d(embed_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_channels),
            nn.ReLU(),
        )

        self.encoder = nn.ModuleList()
        in_channels = embed_channels
        for i in range(self.nblock):
            self.encoder.append(
                PVDSA_Res(
                    in_channels=in_channels,
                    out_channels=out_channels[i],
                    nsample=self.nsample,
                    resolution=resolution[i],
                    with_se=cps.pop('with_se', False),
                    normalize=cps.pop('normalize', False),
                    eps=cps.pop('eps', 0),
                    agg_way=cps.pop('agg_way', 'add'),
                    res=cps.pop('res', True),
                    conv_res=cps.pop('conv_res', False),
                    pvdsa_class=7,
                    **cps
                )
            )
            in_channels = out_channels[i]

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(sum(out_channels), 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.label_conv = nn.Sequential(
            nn.Conv1d(args.num_classes, self.shape_label_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.shape_label_channels),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(1024 * 3 + self.shape_label_channels, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, args.num_classes, 1),
        )
        self.patch_embed1 = PatchEmbed(
            in_chans=63, embed_dim=embed_channels)
        self.embed_fn, self.input_ch = get_embedder(10)
        self.norm1 = nn.BatchNorm1d(embed_channels)

    def forward(self, points, label):
        x = points[:, :self.in_channels, :]
        xyz = x[:, 0:3, :]
        b, _, n = x.size()
        cls_label_one_hot = label.transpose(1, 2).repeat(1, 1, n)#b,class,n
        x = self.embed_fn(xyz.transpose(1, 2)).transpose(1, 2)
        idx = knn_point(self.k, xyz, xyz)
        x = self.patch_embed1(x)
        # input embedding
        x = x+self.input_embedding(xyz)
        # x = self.input_embedding(xyz)

        x = self.norm1(x)
        # feature encoding
        xs = []
        for i in range(self.nblock):
            x, _, _, = self.encoder[i]((x, xyz, idx))
            # x, _, _, cnn,trans = self.encoder[i]((x, xyz, idx))
            xs.append(x)
            # save_branch_features(feat_cnn=cnn,feat_trans=trans,feat_fuse=x,root_dir="./log/part_seg/loop=10/1028_Abliation/Soybean_1028_rm_DGFEB/Soybean_1024-30_down/vis",encoder_layer=i)   #feat vis

        x = torch.cat(xs, dim=1)
        x = self.conv_fuse(x)  # b, 1024, n

        x_max = x.max(dim=-1).values.view(b, -1)  # b, c
        x_avg = x.mean(dim=-1).view(b, -1)  # b, c

        x_max_features = x_max.unsqueeze(-1).repeat(1, 1, n)
        x_avg_features = x_avg.unsqueeze(-1).repeat(1, 1, n)
        cls_label_feature = self.label_conv(cls_label_one_hot)
        x_global_features = torch.cat((x_max_features, x_avg_features, cls_label_feature), 1)  # b, 1024*2+64, n
        x = torch.cat((x, x_global_features), dim=1)  # b, 1024*3+64, n

        x = self.classifier(x)  # b, 50, n
        #return x, None
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1)
        return x


    
class get_model(nn.Module):
    def __init__(self, part_num=50, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            extra_channel = 3
        else:
            extra_channel = 0

        self.pvdsnet = PVDST_partseg(num_classes=args.num_classes, num_shapes=args.num_shapes, extra_feature_channels=extra_channel)

    def forward(self, point_cloud, label):
        x, trans_feat = self.pvdsnet(point_cloud, label)
        x = F.log_softmax(x, dim=1)

        return x.transpose(1, 2).contiguous(), trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    #def forward(self, pred, target, trans_feat):
    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        
        return total_loss





        

    
