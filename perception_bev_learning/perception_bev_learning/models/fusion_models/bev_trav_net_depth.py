from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from perception_bev_learning.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from perception_bev_learning.ops import Voxelization, DynamicScatter
from perception_bev_learning.models.fusion_models import BevTravNet

__all__ = ["BevTravNetDepth"]

# TODO: Accomodate in the same network -> Keep flexible the depth return


class BevTravNetDepth(BevTravNet):
    def __init__(
        self,
        cfg_model: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(cfg_model, **kwargs)

    def forward(
        self,
        imgs,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        target_shape,
        pcd,
        gvom=None,
        aux=None,
    ):
        features = []
        for sensor in self.encoders:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    imgs,
                    rots,
                    trans,
                    intrins,
                    post_rots,
                    post_trans,
                )
                if type(feature) == tuple:
                    feature, depth = feature

            elif sensor == "lidar":
                feature = self.extract_features(pcd, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if self.cfg_model.use_raw_elevation:
            ele_features = torch.nan_to_num(aux[:, self.elevation_idx], 0)[:, None]
            if self.cfg_model.downsample_raw_ele > 1:
                ele_features = torch.nn.functional.interpolate(
                    ele_features,
                    scale_factor=1.0 / self.cfg_model.downsample_raw_ele,
                    mode="bilinear",
                )
            features.append(ele_features)

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        x = self.head(x[0])

        # depth is in shape NxB, D
        return x, depth
