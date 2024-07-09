import torch.nn as nn
import torch
from typing import Any, Dict
from perception_bev_learning.models.builder import build_backbone, build_neck
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class PointPillarsNet(nn.Module):
    def __init__(
        self,
        voxel_encoder: Dict[str, Any],
        middle_encoder: Dict[str, Any],
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.voxel_encoder = build_backbone(voxel_encoder)
        self.middle_encoder = build_backbone(middle_encoder)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)

    def forward(self, feats, coords, batch_size, sizes):
        x = self.voxel_encoder(feats, sizes, coords)
        x = self.middle_encoder(x, coords, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x[0]
