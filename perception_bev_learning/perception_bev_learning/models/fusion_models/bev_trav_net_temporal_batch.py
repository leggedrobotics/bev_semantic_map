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
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import affine

__all__ = ["BevTravNetTemporalBatch"]


class BevTravNetTemporalBatch(nn.Module):
    def __init__(
        self,
        cfg_model: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.cfg_model = cfg_model
        self.encoders = nn.ModuleDict()
        encoders = OmegaConf.to_container(cfg_model["encoders"])
        fuser = OmegaConf.to_container(cfg_model.get("fuser", None))
        decoder = OmegaConf.to_container(cfg_model["decoder"])
        head = OmegaConf.to_container(cfg_model["head"])
        self.sequence_length = self.cfg_model.sequence_length
        fuser_input_channels = []
        self.fuser_output_channels = fuser["out_channels"]
        self.gmr_resolution = self.cfg_model.gmr_resolution

        if cfg_model.use_images:
            if encoders.get("camera") is not None:
                self.encoders["camera"] = nn.ModuleDict(
                    {
                        "backbone": build_backbone(encoders["camera"]["backbone"]),
                        "neck": build_neck(encoders["camera"]["neck"]),
                        "vtransform": build_vtransform(
                            encoders["camera"]["vtransform"]
                        ),
                    }
                )
                fuser_input_channels.append(
                    cfg_model.encoders.camera.vtransform.out_channels
                )
            else:
                raise ValueError(
                    f"Configured to use images but image encoder config is missing !"
                )
        if cfg_model.use_lidar:
            if encoders.get("lidar") is not None:
                if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                    voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
                else:
                    voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
                self.encoders["lidar"] = nn.ModuleDict(
                    {
                        "voxelize": voxelize_module,
                        "backbone": build_backbone(encoders["lidar"]["backbone"]),
                    }
                )
                self.voxelize_reduce = encoders["lidar"].get(
                    "voxelize_reduce", False
                )  # TODO verify and not implemented for the output channels consistency
                fuser_input_channels.append(
                    sum(cfg_model.encoders.lidar.backbone.neck.out_channels)
                )
            else:
                raise ValueError(
                    f"Configured to use lidar but lidar encoder config is missing !"
                )

        if cfg_model.use_gvom:
            if encoders.get("gvom") is not None:
                if encoders["gvom"]["voxelize"].get("max_num_points", -1) > 0:
                    voxelize_module = Voxelization(**encoders["gvom"]["voxelize"])
                else:
                    voxelize_module = DynamicScatter(**encoders["gvom"]["voxelize"])
                self.encoders["gvom"] = nn.ModuleDict(
                    {
                        "voxelize": voxelize_module,
                        "backbone": build_backbone(encoders["gvom"]["backbone"]),
                    }
                )
                fuser_input_channels.append(
                    sum(cfg_model.encoders.gvom.backbone.neck.out_channels)
                )
            else:
                raise ValueError(
                    f"Configured to use gvom cloud but config is missing !"
                )

        if cfg_model.use_raw_elevation:
            fuser_input_channels.append(1)
            self.elevation_idx = [
                j
                for j, l in enumerate(cfg_model.aux_layers.values())
                if (l.name == "elevation_raw")
            ][0]

        fuser["in_channels"] = fuser_input_channels
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        # Recurrent Temporal Module
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(
                (fuser["out_channels"]) * 2,
                fuser["out_channels"],
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fuser["out_channels"]),
            nn.ReLU(),
        )

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        self.head = build_head(head)
        self.init_weights()

        if cfg_model.freeze_backbones:
            for param in self.encoders["camera"].parameters():
                param.requires_grad = False
            for param in self.encoders["lidar"].parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self, x, rots, trans, intrins, post_rots, post_trans, *args, **kwargs
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.reshape(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.reshape(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
        )
        return x

    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x["points"], x["batch"], sensor)
        batch_size = coords[-1, 0] + 1
        # print(feats.shape, coords.shape, sizes.shape, batch_size)
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, batch, sensor):
        batched_pts = []
        start = 0
        for b in range(batch.shape[0]):
            batched_pts.append(points[start : batch[b] + start])
            start += int(batch[b])

        feats, coords, sizes = [], [], []
        for k, res in enumerate(batched_pts):
            res = res.cuda()
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def reset_temporal(self, aux):
        """
        Reset the temoral state
        """
        B, C, H, W = aux.shape
        self.H_c = H // 2
        self.W_c = W // 2
        self.bev_features_hidden = torch.zeros(
            [B, self.fuser_output_channels, H, W],
            device=aux.device,
            dtype=aux.dtype,
            requires_grad=False,
        )

        self.prev_H_sg_map = (
            torch.eye(4, device=aux.device, dtype=aux.dtype, requires_grad=False)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

    def compensate_ego_motion(self, H_sg_map):
        B, _, _ = H_sg_map.shape
        grid_map_data_prev_corrected = []

        for i in range(B):
            H_sg_sgprev = H_sg_map[i] @ torch.inverse(self.prev_H_sg_map[i])
            ypr = R.from_matrix(H_sg_sgprev[:3, :3].cpu().numpy()).as_euler(
                seq="zyx", degrees=True
            )
            shift = (H_sg_sgprev[:2, 3]) / self.gmr_resolution
            sh = [shift[1], shift[0]]
            yaw = ypr[0]
            # print("sh ", i, list(sh))
            # print("yaw ", i, yaw)
            grid_map_data_corrected = affine(
                self.bev_features_hidden[i][None],
                angle=-yaw,
                translate=list(sh),
                scale=1,
                shear=0,
                center=(self.H_c, self.W_c),
                fill=0,
            )[0]
            grid_map_data_prev_corrected.append(grid_map_data_corrected)

        self.bev_features_hidden = torch.stack(grid_map_data_prev_corrected)

    # @auto_fp16(apply_to=("img", "points"))
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
        H_sg_map=None,
        batch_idx=None,
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

            elif sensor == "lidar":
                feature = self.extract_features(pcd, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if self.cfg_model.use_raw_elevation:
            features.append(torch.nan_to_num(aux[:, self.elevation_idx], 0)[:, None])

        if self.fuser is not None:
            fused_output = self.fuser(features)
        else:
            assert len(features) == 1, features
            fused_output = features[0]

        ##############################################################################
        if self.cfg_model.use_temporal_fusion:
            self.compensate_ego_motion(H_sg_map)
            self.bev_features_hidden = self.temporal_fusion(
                torch.cat([fused_output, self.bev_features_hidden], dim=1)
            )
            self.prev_H_sg_map = H_sg_map
        else:
            self.bev_features_hidden = fused_output

        x = self.decoder["backbone"](self.bev_features_hidden)
        x = self.decoder["neck"](x)
        x = self.head(x[0])

        return x
