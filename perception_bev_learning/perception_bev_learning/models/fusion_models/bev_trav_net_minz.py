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
from perception_bev_learning.models.backbones import PointPillarsScatter
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from perception_bev_learning.ops import voxelize_pcd_scans

__all__ = ["BevTravNetMinZ"]

CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ELEVATION.set_bad(color="black")


class BevTravNetMinZ(nn.Module):
    def __init__(self, cfg_model: Dict[str, Any], **kwargs) -> None:
        super().__init__()

        self.cfg_model = cfg_model
        self.encoders = nn.ModuleDict()
        encoders = OmegaConf.to_container(cfg_model["encoders"])
        fuser = OmegaConf.to_container(cfg_model.get("fuser", None))
        decoder = OmegaConf.to_container(cfg_model["decoder"])
        head = OmegaConf.to_container(cfg_model["head"])
        fuser_input_channels = []

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
                self.min_z_scatter = PointPillarsScatter(
                    in_channels=3,
                    output_shape=encoders["lidar"]["backbone"]["middle_encoder"][
                        "output_shape"
                    ],
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
            x, rots, trans, intrins, post_rots, post_trans
        )
        return x

    def extract_features(self, x, sensor) -> torch.Tensor:
        print(x["points"].shape)
        feats, coords, sizes = self.voxelize(x["points"], x["batch"], sensor)
        batch_size = coords[-1, 0] + 1
        # print(feats.shape, coords.shape, sizes.shape, batch_size)
        # pcd_data = voxelize_pcd_scans(
        #     x["points"].clone().detach(),
        #     x["batch"].clone().detach(),
        #     x["scan"].clone().detach(),
        #     (512, 512, 1),
        #     (0.2, 0.2, 20),
        # )
        # print(f"pcd data shape is {pcd_data.shape}")
        # print(f"pcd data sum is {torch.sum(pcd_data != 0)}")
        # Extract the minimum Z
        z_values = feats[:, :, 2]
        zero_mask = z_values == 0
        non_zero_z_values = z_values.masked_fill(zero_mask, 10000)

        # print(non_zero_z_values)
        min_z_indices = torch.argmin(non_zero_z_values, dim=1)
        # print(min_z_indices)
        # print(f"min z ind {min_z_indices.shape}")
        min_z_features = feats[torch.arange(feats.size(0)), min_z_indices, :]
        print(feats.shape, coords.shape, sizes.shape, batch_size)
        print("Before scatter ", torch.sum(feats[:, 2] != 0))
        print(f"min z scatter {min_z_features.shape}")
        # Move them to BevGrid
        min_z_feat = self.min_z_scatter(min_z_features, coords, batch_size)
        print("min_z_feat.shape ", min_z_feat.shape)
        print("After scatter ", torch.sum(min_z_feat[:, 2] != 0))
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x, min_z_feat, None

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

            # Generate shuffled indices
            num_points = res.shape[0]
            shuffled_indices = torch.randperm(num_points)
            res = res[shuffled_indices]

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
        batch_idx=None,
    ):
        features = []
        for sensor in self.encoders:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    imgs, rots, trans, intrins, post_rots, post_trans
                )

            elif sensor == "lidar":
                feature, min_z_feat, pcd_data = self.extract_features(pcd, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        # Plot the Raw elevation and Pointcloud Values:
        ele_feat_raw = aux[:, self.elevation_idx][:, None]
        min_z_feat = min_z_feat[:, 2]
        # min_z_feat[min_z_feat != 0] = 1
        print(torch.sum(min_z_feat != 0))
        # min_z_feat[min_z_feat == 0] = -9
        print(min_z_feat.shape)

        min_z_feat[min_z_feat >= 5000] = 0
        min_z_feat[min_z_feat == 0] = -9
        mask = ~(torch.isnan(min_z_feat)) * ~(torch.isnan(ele_feat_raw))

        raw_pcl_ele = torch.flip(min_z_feat.detach().cpu(), dims=(1, 2))
        raw_curr_ele = torch.flip(ele_feat_raw.detach().cpu()[0], dims=(1, 2)) * 20
        error_ele = torch.abs(raw_pcl_ele - raw_curr_ele)
        # maps = torch.stack(
        #     [raw_pcl_ele, raw_curr_ele, torch.flip(pcd_data.detach().cpu()[0,0], dims=(1,2))],
        #     dim=0,
        # )
        maps = torch.stack([raw_pcl_ele, raw_curr_ele, error_ele], dim=0)
        color_maps = [CMAP_ELEVATION] * 3
        N = maps.shape[0]
        titles = ["pcl raw ele", "current raw ele", "Error"]
        v_mins = [-10] * 2 + [0] * 1
        v_maxs = [10] * 2 + [5] * 1
        nrows_ncols = (1, N)
        maps = maps.type(torch.float32).numpy()
        fig = plt.figure(figsize=(nrows_ncols[1] * 4.0 - 3, nrows_ncols[0] * 4.0))
        fig.tight_layout(pad=1.1)
        grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

        for i in range(N):
            v_min = (
                maps[i][~np.isnan(maps[i])].min() if v_mins[i] is None else v_mins[i]
            )
            v_max = (
                maps[i][~np.isnan(maps[i])].max() if v_maxs[i] is None else v_maxs[i]
            )

            grid[i].imshow(maps[i][0], cmap=color_maps[i], vmin=v_min, vmax=v_max)
            grid[i].set_title(titles[i], fontsize=16)
            grid[i].grid(False)

        fig.tight_layout(pad=0.0)
        plt.savefig(f"/data_ssd/manthan/results/image_pcl_minz/{batch_idx}.png")
        plt.close()

        ###############################################

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

        return x

        # output = {
        #     "wheel_risk": [],
        #     "elevation": [],
        # }
        # for i in range(50):
        #     y = self.decoder["backbone"](x)
        #     y = self.decoder["neck"](y)
        #     y = self.head(y[0])
        #     output["wheel_risk"].append(y["wheel_risk"])
        #     output["elevation"].append(y["elevation"])

        # return output
