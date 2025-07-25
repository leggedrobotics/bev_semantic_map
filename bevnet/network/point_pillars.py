import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
from .generic_pointcloud_backbone import GenericPointcloudBackbone
from pytictac import accumulate_time
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import Voxelization

from icecream import ic


class PillarLayer(nn.Module):
    # TODO check lincense: https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        """
        batched_pts: list[tensor], len(batched_pts) = bs
        return:
            pillars: (p1 + p2 + ... + pb, num_points, c),
            coors_batch: (p1 + p2 + ... + pb, 1 + 3),
            num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))

        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    # TODO check lincense: https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = (
            pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]
        )  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (
            coors_batch[:, None, 1:2] * self.vx + self.x_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
            coors_batch[:, None, 2:3] * self.vy + self.y_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat(
            [pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1
        )  # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        # ic(features.shape)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        # ic(features.shape)
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels), out_channels = 64
        # ic(pooling_features.shape)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=cur_features.dtype, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    # TODO check lincense: https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    # TODO check lincense: https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(
                nn.ConvTranspose2d(
                    in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False
                )
            )
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class PointPillars(GenericPointcloudBackbone):
    # TODO check lincense: https://github.com/zhulf0804/PointPillars/blob/main/model/pointpillars.py
    # mostly pasted at the moment
    def __init__(self, cfg):
        """Description

        Args:
            cfg (PointPillarsParams): Configuration
        """
        super(PointPillars, self).__init__()
        self.cfg = cfg
        self.pillar_layer = PillarLayer(
            voxel_size=cfg.voxel_size,
            point_cloud_range=cfg.point_cloud_range,
            max_num_points=cfg.max_num_points,
            max_voxels=cfg.max_voxels,
        )

        self.pillar_encoder = PillarEncoder(
            voxel_size=cfg.voxel_size, point_cloud_range=cfg.point_cloud_range, in_channel=8, out_channel=64
        )

        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])

        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[32, 32, 32])

    @accumulate_time
    def preprocessing(self, x: torch.Tensor, batch: torch.Tensor, scan: torch.Tensor):
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)

        batched_pts = []
        start = 0
        for b in range(batch.shape[0]):
            batched_pts.append(x[start : batch[b] + start])
            start += int(batch[b])

        # WARNING: When creating the Pilars most likely a different coordinate system convention is used
        # For us it was
        #  ------ y   Where x is the forward direction and y is to the left of the vehicle
        # |           When we plot the maps we flip the map such that x is up and left is on the left of the gridmap
        # |
        # |
        # x
        # Here I assumes the coordinate indiced are flipped x=y and y=x
        # Given that map is square this should not matter and we just need to flip the x and y axis before returning features
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        # Convert Pointcloud to PointPillars3D
        return {"pillars": pillars, "coors_batch": coors_batch, "npoints_per_pillar": npoints_per_pillar}

    @accumulate_time
    def embed(self, x):
        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        # -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(x["pillars"], x["coors_batch"], x["npoints_per_pillar"])
        feat = self.backbone(pillar_features)
        feat = self.neck(feat)

        return feat.permute(0, 1, 3, 2)  # Reshape the map to normal coordinate system

    @accumulate_time
    def postprocessing(self, x):
        return x
