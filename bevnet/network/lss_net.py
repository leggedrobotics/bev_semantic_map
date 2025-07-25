"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from pytictac import Timer
from .lss_tools import gen_dx_bx, cumsum_trick, QuickCumsum
from bevnet.ops import bev_pool
from icecream import ic


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        # print("x1", x1.shape)
        # print("x2", x2.shape)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x, *args, **kwargs):
        """
        Args:
            x (torch.tensor, shape=(BS, NR_CAM, C, H, W), dtype=torch.float32):
        """
        x = self.get_eff_depth(x)
        # Depth
        # BS*NR_CAM x C x H/16 x W/16
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):

        # BS,
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x (torch.tensor, shape=(BS, NR_CAM, C, H, W), dtype=torch.float32):
        """
        depth, x = self.get_depth_feat(x, *args, **kwargs)
        return x


class MultiHeadBevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(MultiHeadBevEncode, self).__init__()

        heads = []
        for j in range(outC):
            heads.append(BevEncode(inC, 1))

        self.heads = nn.ModuleList(heads)  # Puts modules in a list

    def forward(self, x):
        res = []
        for head in self.heads:
            res.append(head(x))
        return torch.cat(res, dim=1)


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # ic(x.shape)
        x = self.conv1(x)
        # ic(x.shape)
        x = self.bn1(x)
        # ic(x.shape)
        x = self.relu(x)
        # ic(x.shape)

        x1 = self.layer1(x)
        # ic(x1.shape)
        x = self.layer2(x1)
        # ic(x.shape)
        x = self.layer3(x)
        # ic(x.shape)

        x = self.up1(x, x1)
        # ic(x.shape)
        x = self.up2(nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
        # ic(x.shape)

        return x


class LiftSplatShootNet(nn.Module):
    def __init__(self, cfg):
        """Network

        Args:
            cfg (LiftSplatShootNetParams): _description_
        """
        super(LiftSplatShootNet, self).__init__()
        self.grid_conf = cfg.grid
        self.data_aug_conf = cfg.augmentation

        dx, bx, nx = gen_dx_bx(self.grid_conf.xbound, self.grid_conf.ybound, self.grid_conf.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        if cfg.bevencode:
            self.bevencode = BevEncode(inC=self.camC, outC=cfg.output_channels)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        self.use_quickcumsum_cuda = True

    def create_frustum(self, save_frustrum=False):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf.fH, self.data_aug_conf.fW
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # Frustrum in image plane

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, *args, **kwargs):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)

        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, *args, **kwargs):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)  # BN x C x H x W, bring in right shape for efficientnet
        # print(x.shape)
        x = self.camencode(x)
        # print(x.shape)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        # print(x.shape)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # print(x.shape)
        return x

    def voxel_pooling(self, geom_feats, x, *args, **kwargs):
        B, N, D, H, W, C = x.shape

        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if self.use_quickcumsum_cuda:  # Very fast
            # print("Using QuickCumsum CUDA")
            # self.nv account for BEV gridmap size H x W x 1?
            x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
            final = torch.cat(x.unbind(dim=2), 1)
            return final
        elif not self.use_quickcumsum:  # Slow
            # print("Using QuickCumsum Python")
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:  # Fast
            # print("Using QuickCumsum")
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device, dtype=x.dtype)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        # final shape: B x C x GRID_X x GRID_Y
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, *args, **kwargs):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, *args, **kwargs)

        # Save the frustrum for debugging
        # torch.save(geom, "/home/rschmid/RosBags/bevnet2/others/frustrum.pt")

        x = self.get_cam_feats(x, *args, **kwargs)  # Splatting features
        x = self.voxel_pooling(geom, x, *args, **kwargs)  # Projecting on 2d BEV grid
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, *args, **kwargs):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, *args, **kwargs)
        if hasattr(self, "bevencode"):  # Set to false atm
            x = self.bevencode(x)
        return x
