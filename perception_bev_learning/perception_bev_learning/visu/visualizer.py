# TODO: Jonas doc strings, rework visualiation functions

import os
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
import torch
import skimage
import seaborn as sns
import pytorch_lightning as pl
from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import open3d as o3d
from os.path import join
from perception_bev_learning.visu import image_functionality
from perception_bev_learning.visu import get_img_from_fig
from perception_bev_learning.visu import (
    paper_colors_rgb_u8,
    paper_colors_rgba_u8,
    racer_risk_colormap,
    unknown_color,
)
from perception_bev_learning.visu import paper_colors_rgb_f, paper_colors_rgba_f
from perception_bev_learning.visu import DrawPolygonHelper
import cv2
from perception_bev_learning.utils import Timer
from perception_bev_learning.utils import denormalize_img
from torch import Tensor
from typing import List, Optional, Tuple
from perception_bev_learning.utils import invert_se3
from perception_bev_learning.ops import voxelize_pcd_scans

__all__ = ["LearningVisualizer"]


class LearningVisualizer:
    def __init__(
        self,
        p_visu: Optional[str] = None,
        store: Optional[bool] = False,
        pl_model: Optional[pl.LightningModule] = None,
        epoch: int = 0,
        log: bool = False,
    ):
        self._p_visu = p_visu
        self._pl_model = pl_model
        self._epoch = epoch
        self._store = store
        self._log = log
        self._c_maps = {
            "RdYlBu": np.array(
                [np.uint8(np.array(c) * 255) for c in sns.color_palette("RdYlBu", 256)]
            )
        }

        self.draw_polygon_helper = DrawPolygonHelper(594, 960)

        if not (p_visu is None):
            if not os.path.exists(self._p_visu):
                try:
                    os.makedirs(self._p_visu)
                except OSError:
                    print("Creation of the directory %s failed" % self._p_visu)
        else:
            self._store = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def store(self) -> bool:
        return self._store

    @store.setter
    def store(self, store: bool):
        self._store = store

    @image_functionality
    def plot_list(self, imgs, **kwargs):
        return np.concatenate(imgs, axis=1)

    @image_functionality
    def plot_elevation_map(
        self,
        elevation,
        reverse=True,
        elevation_map=False,
        v_min=None,
        v_max=None,
        cmap_name=None,
        **kwargs,
    ):
        # Elevation 1,N_Cells,M_Cells torch.tensor

        if type(elevation) == torch.Tensor:
            elevation = elevation.cpu().numpy()
        elevation_mask = ~np.isnan(elevation)

        if v_min is None:
            v_min = elevation[elevation_mask].min()
        if v_min is None:
            v_max = elevation[elevation_mask].max()

        elevation = np.ma.masked_where(~elevation_mask, elevation)

        if cmap_name is not None:
            cmap = sns.color_palette(cmap_name, as_cmap=True)
        else:
            if reverse:
                cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
            else:
                cmap = sns.color_palette("RdYlBu", as_cmap=True)

            if elevation_map:
                cmap = sns.color_palette("viridis", as_cmap=True)

        cmap.set_bad(color="black")

        fig = plt.figure(
            figsize=(
                int(0.03 * elevation.shape[0] + 1),
                int(0.03 * elevation.shape[1] + 1),
            )
        )
        plt.imshow(elevation, cmap=cmap, vmin=v_min, vmax=v_max)
        plt.title(f"Elevation", fontsize=16)
        fig.tight_layout(pad=0.0)
        res = get_img_from_fig(fig)
        plt.close()
        return np.array(res)

    @image_functionality
    def plot_elevation_map_two(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        scale_target: bool = True,
        reverse: bool = True,
        v_min=None,
        v_max=None,
        auto_scale: bool = False,
        cmap_name=None,
        labels=["target", "pred"],
        subtitle=True,
        store_svg=False,
        **kwargs,
    ):
        target = torch.flip(target, [0, 1])
        pred = torch.flip(pred, [0, 1])

        target_mask = ~torch.isnan(target)
        pred_mask = torch.ones_like(pred).type(torch.bool)

        mi1 = torch.quantile(target[target_mask], 0)
        ma1 = torch.quantile(target[target_mask], 1)
        mi2 = torch.quantile(pred[pred_mask].type(torch.float32), 0)
        ma2 = torch.quantile(pred[pred_mask].type(torch.float32), 1)

        if type(target) == torch.Tensor:
            target = target.cpu().numpy()
            target_mask = target_mask.cpu().numpy()
        if type(pred) == torch.Tensor:
            pred = pred.type(torch.float32).cpu().numpy()
            pred_mask = pred_mask.cpu().numpy()

        if scale_target:
            mi = mi1
            ma = ma1
        else:
            mi = min(mi1, mi2)
            ma = max(ma1, ma2)

        if v_min is not None:
            mi = v_min
        if v_max is not None:
            ma = v_max

        nrows_ncols = (1, 2)
        fig, axs = plt.subplots(
            1, 2, figsize=(nrows_ncols[1] * 4.0 + 2, nrows_ncols[0] * 4.0 + 2)
        )

        if cmap_name is not None:
            cmap = sns.color_palette(cmap_name, as_cmap=True)
        else:
            if reverse:
                cmap = sns.color_palette("RdYlBu_r", as_cmap=True)
            else:
                cmap = sns.color_palette("RdYlBu", as_cmap=True)

        cmap.set_bad(color="black")

        if target_mask is not None:
            target = np.ma.masked_where(~target_mask, target)
        if pred_mask is not None:
            pred = np.ma.masked_where(~pred_mask, pred)

        if auto_scale:
            mi, ma = None, None

        im0 = axs[0].imshow(target, cmap=cmap, vmin=mi, vmax=ma)
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title(labels[0])
        axs[0].grid(False)

        im1 = axs[1].imshow(pred, cmap=cmap, vmin=mi, vmax=ma)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title(labels[1])
        axs[1].grid(False)

        if subtitle:
            fig.suptitle(f"Min: {mi}, Max: {ma}")
        fig.tight_layout()

        if store_svg:
            folder = self._p_visu
            tag = kwargs.get("tag", "")
            plt.savefig(join(folder, f"{self.epoch}_{tag}.svg"), format="svg")

        img = get_img_from_fig(fig)

        plt.close()
        return np.array(img)

    @image_functionality
    def plot_n_maps(
        self,
        maps: torch.Tensor,
        color_maps: list,
        titles: list,
        v_mins: list,
        v_maxs: list,
        store_svg: bool = False,
        **kwargs,
    ):
        maps = maps.type(torch.float32).cpu().numpy()[:, ::-1, ::-1]
        N = maps.shape[0]
        nrows_ncols = (1, N)
        fig = plt.figure(figsize=(nrows_ncols[1] * 4.0 - 3, nrows_ncols[0] * 4.0))
        fig.tight_layout(pad=1.1)
        grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

        for i in range(N):
            cm = color_maps[i]

            v_min = (
                maps[i][~np.isnan(maps[i])].min() if v_mins[i] is None else v_mins[i]
            )
            v_max = (
                maps[i][~np.isnan(maps[i])].max() if v_maxs[i] is None else v_maxs[i]
            )

            grid[i].imshow(maps[i], cmap=color_maps[i], vmin=v_min, vmax=v_max)
            grid[i].set_title(titles[i], fontsize=16)
            grid[i].grid(False)

        fig.tight_layout(pad=0.0)

        if store_svg:
            tag = kwargs.get("tag", "")
            plt.savefig(join(self._p_visu, f"{self.epoch}_{tag}.svg"), format="svg")

        img = get_img_from_fig(fig)
        plt.close()
        return np.array(img)

    @image_functionality
    def plot_all_maps(
        self,
        pred: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        reliable: torch.Tensor,
        pcd_maps: torch.Tensor,
        gvom_maps: torch.Tensor,
        elevation_pred: None,
        elevation_current: None,
        elevation_target: None,
        scale_target: bool = True,
        reverse: bool = True,
        v_min: float = 0.0,
        v_max: float = 1.0,
        store_svg: bool = False,
        **kwargs,
    ):
        if type(pred) == torch.Tensor:
            pred = pred.type(torch.float32).cpu().numpy()[::-1, ::-1]
        if type(current) == torch.Tensor:
            current = current.type(torch.float32).cpu().numpy()[::-1, ::-1]
        if type(target) == torch.Tensor:
            target = target.cpu().numpy()[::-1, ::-1]

        if type(reliable) == torch.Tensor:
            reliable = reliable.type(torch.float32).cpu().numpy()[::-1, ::-1]

        elevations = [elevation_pred, elevation_current, elevation_target]
        for i in range(3):
            if type(elevations[i]) == torch.Tensor:
                elevations[i] = (
                    elevations[i].type(torch.float32).cpu().numpy()[::-1, ::-1]
                )
                plot_ele = True

        target_mask = ~np.isnan(target)
        pred_mask = np.ones_like(pred).astype(bool)
        current_mask = ~np.isnan(current)
        reliable_mask = ~np.isnan(reliable)

        nrows_ncols = (2, 6)
        fig = plt.figure(figsize=(nrows_ncols[1] * 4.0 - 3, nrows_ncols[0] * 4.0))
        fig.tight_layout(pad=0.0)
        grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

        cmap_traversability = sns.color_palette("RdYlBu_r", as_cmap=True)
        cmap_reliable = sns.color_palette("viridis", as_cmap=True)
        cmap_elevation = sns.color_palette("viridis", as_cmap=True)
        cmap_error = sns.color_palette("rocket", as_cmap=True)

        # cmap_traversability = racer_risk_colormap

        cmap_traversability.set_bad(color="black")
        cmap_reliable.set_bad(color="black")
        cmap_elevation.set_bad(color="black")
        cmap_error.set_bad(color="black")

        target = np.ma.masked_where(~target_mask, target)
        pred = np.ma.masked_where(~pred_mask, pred)
        current = np.ma.masked_where(~current_mask, current)
        reliable = np.ma.masked_where(~reliable_mask, reliable)

        col = [np.array(v) for v in paper_colors_rgb_u8.values()]
        # Set nan to 0
        # print(f"GVOM shape is {gvom_maps.shape}")
        # print(f"PCD shape is {pcd_maps.shape}")

        pcd_maps = torch.flip(pcd_maps, [1, 2])
        gvom_maps = torch.flip(gvom_maps, [1, 2])

        if not (gvom_maps.shape[0] == 1 and gvom_maps.shape[1] == 1):
            fused_img = np.zeros(
                (gvom_maps[0].shape[0], gvom_maps[0].shape[1], 3), dtype=np.uint8
            )
            for j, _map in enumerate(gvom_maps):
                _map = _map.cpu().numpy()
                fused_img[_map] = col[j]

            grid[1].imshow(fused_img)
            grid[1].set_title(f"GVOM Voxel Cloud", fontsize=16)

        if not (pcd_maps.shape[0] == 1 and pcd_maps.shape[1] == 1):
            # for _map in pcd_maps:
            #     _map[torch.isnan(_map)] = 0
            # # Make binary
            # pcd_maps = pcd_maps != 0

            fused_img = np.zeros(
                (pcd_maps[0].shape[0], pcd_maps[0].shape[1], 3), dtype=np.uint8
            )
            for j, _map in enumerate(pcd_maps):
                _map = _map.cpu().numpy()
                fused_img[_map] = col[j]

            grid[4].imshow(fused_img)
            grid[4].set_title(f"N={pcd_maps.shape[0]} Accu-Pointclouds", fontsize=16)

        grid[0].imshow(pred, cmap=cmap_traversability, vmin=v_min, vmax=v_max)
        grid[0].set_title(f"Risk - Pred", fontsize=16)
        grid[0].grid(False)
        grid[1].imshow(current, cmap=cmap_traversability, vmin=v_min, vmax=v_max)
        grid[1].set_title(f"Risk - RACER-X", fontsize=16)
        grid[1].grid(False)

        im0 = grid[2].imshow(target, cmap=cmap_traversability, vmin=v_min, vmax=v_max)
        grid[2].set_title(f"Risk - GT", fontsize=16)
        grid[2].grid(False)
        grid[3].imshow(reliable, cmap=cmap_reliable, vmin=v_min, vmax=v_max)
        grid[3].set_title(f"Reliability - RACER-X", fontsize=16)
        grid[3].grid(False)
        nr = 5
        dynamic = False
        for e, text in zip(elevations, ["Elev - Pred", "Elev - RACER-X", "Elev - GT"]):
            if dynamic:
                v_min = max(
                    np.array(
                        [e[~np.isnan(e)].min() for e in elevations if e is not None]
                    ).min(),
                    -20,
                )
                v_max = min(
                    np.array(
                        [e[~np.isnan(e)].max() for e in elevations if e is not None]
                    ).max(),
                    20,
                )
            else:
                v_min = -20
                v_max = 20
            if e is not None:
                im_last = grid[nr].imshow(
                    e, cmap=cmap_elevation, vmin=v_min, vmax=v_max
                )
                grid[nr].set_title(text, fontsize=16)
                grid[nr].grid(False)
                nr += 1

        fig.tight_layout(pad=0.0)

        if store_svg:
            folder = self._p_visu
            tag = kwargs.get("tag", "")
            plt.savefig(join(folder, f"{self.epoch}_{tag}.svg"), format="svg")

        img = get_img_from_fig(fig)
        plt.close()
        return np.array(img)

    @image_functionality
    def plot_pcd_bev(self, maps: torch.tensor, binary: bool = True, **kwargs):
        """

        Args:
            maps (torch.tensor): N, H, W torch.float32
            binary (False): _description_

        Returns:
            _type_: _description_
        """
        assert binary == True, "Not implemented"

        col = [np.array(v) for v in paper_colors_rgb_u8.values()]

        # Set nan to 0
        for _map in maps:
            # Mirror
            _map = torch.flip(_map, [0, 1])
            # _map = _map[:, ::-1, ::-1]
            _map[torch.isnan(_map)] = 0

        # Make binary
        if binary:
            maps = maps != 0

        fused_img = np.zeros((maps[0].shape[0], maps[0].shape[1], 3), dtype=np.uint8)
        single_imgs = []

        for j, _map in enumerate(maps):
            single_imgs.append(
                np.zeros((_map.shape[0], _map.shape[1], 3), dtype=np.uint8)
            )
            _map = _map.cpu().numpy()
            single_imgs[-1][_map] = col[j]
            fused_img[_map] = col[j]

        # Plotting
        imgs = single_imgs + [fused_img]
        if len(imgs) < 17:
            nrows_ncols = (4, 4)
        if len(imgs) < 10:
            nrows_ncols = (3, 3)
        if len(imgs) < 7:
            nrows_ncols = (2, 3)
        if len(imgs) < 5:
            nrows_ncols = (2, 2)
        if len(imgs) < 3:
            nrows_ncols = (1, 2)

        fig = plt.figure(figsize=(nrows_ncols[1] * 4.0, nrows_ncols[0] * 4.0))

        grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)
        for j, im in enumerate(imgs):
            ax = grid[j]
            ax.imshow(im)
            if j != len(imgs) - 1:
                ax.set_title(f"Single Scan t-{j}", fontsize=16)
            else:
                ax.set_title("Fused Scans", fontsize=16)
        fig.tight_layout(pad=0.0)
        img = get_img_from_fig(fig)
        plt.close()
        return np.array(img)

    @image_functionality
    def project_points_on_image(
        self,
        imgs: torch.tensor,
        rots: torch.tensor,
        trans: torch.tensor,
        intrins: torch.tensor,
        points: torch.tensor,
        cam: int = 0,
        z_color: bool = False,
        **kwargs,
    ):
        # Create 3D points from a given gridmap
        device = imgs.device
        # Get transformation
        H_sensor_gravity__camera = torch.eye(4, device=device)
        H_sensor_gravity__camera[:3, :3] = rots[cam]
        H_sensor_gravity__camera[:3, 3] = trans[cam]
        H_camera__sensor_gravity = torch.inverse(H_sensor_gravity__camera)

        if z_color:
            tar_values = (points[:, 2]).clone().reshape(-1, 1).clip(-6, 1)

        else:
            tar_values = torch.linalg.norm(points[:, :3] - trans[cam], dim=1).reshape(
                -1, 1
            )
            m = ~torch.isnan(tar_values)
            tar_values[m] = tar_values[m].clip(2, 20)
        m = ~torch.isnan(tar_values)
        tar_values[m] = tar_values[m] - tar_values[m].min()
        if tar_values[m].max() != 0:
            tar_values[m] = tar_values[m] / tar_values[m].max()
        tar_values[m] = tar_values[m] * 254

        # Project from sensor_gravity into camera
        uv = (intrins[cam] @ H_camera__sensor_gravity[:3, :] @ points.T).T
        mask_in_front_of_camera = uv[:, 2] > 0
        uv = uv[:, :2] / uv[:, 2, None]
        uv = uv.type(torch.long)
        H, W = imgs[cam].shape[1], imgs[cam].shape[2]

        mask_valid_height = ~torch.isnan(tar_values)[:, 0]

        mask_inside_frustrum = (
            (uv[:, 0] >= 0) * (uv[:, 1] >= 0) * (uv[:, 0] < W) * (uv[:, 1] < H)
        )
        m = mask_inside_frustrum * mask_valid_height * mask_in_front_of_camera
        tar_values = tar_values[m]
        uv = uv[m]

        # Prepare foreground and background image
        background = denormalize_img(imgs[cam])
        background.convert("RGBA")
        foreground = np.array(background).copy()
        foreground[:, :, :] = 0
        foreground = Image.fromarray(foreground).convert("RGBA")
        foreground.putalpha(0)
        draw = ImageDraw.Draw(foreground)
        w = 3
        uv = uv.cpu().tolist()
        for i in range(len(uv)):
            fill = (self._c_maps["RdYlBu"][255 - int(tar_values[i])]).tolist() + [130]
            draw.arc(
                [(uv[i][0], uv[i][1]), (uv[i][0] + w, uv[i][1] + w)],
                start=0,
                end=360,
                fill=tuple(fill),
                width=w,
            )

        background.paste(foreground, (0, 0), foreground)
        return np.array(background)

    @image_functionality
    def project_pcd_on_image(
        self,
        imgs: torch.tensor,
        rots: torch.tensor,
        trans: torch.tensor,
        intrins: torch.tensor,
        pcd_data: torch.tensor,
        grid_map_resolution: float = 0.5,
        cam: int = 0,
        **kwargs,
    ):
        # Create 3D points from a given gridmap
        device = imgs.device
        N = int(pcd_data.shape[1] / 2)
        x_ = torch.arange(-N, N, 1, device=device) * grid_map_resolution
        y_ = torch.arange(-N, N, 1, device=device) * grid_map_resolution
        xv, yv = torch.meshgrid(x_, y_, indexing="ij")
        ones = torch.ones_like(xv)
        points = torch.stack([xv, yv, pcd_data, ones], dim=2).reshape(-1, 4)

        # Get transformation
        H_sensor_gravity__camera = torch.eye(4, device=device)
        H_sensor_gravity__camera[:3, :3] = rots[cam]
        H_sensor_gravity__camera[:3, 3] = trans[cam]
        H_camera__sensor_gravity = torch.inverse(H_sensor_gravity__camera)

        tar_values = pcd_data.reshape(-1, 1)
        m = ~torch.isnan(tar_values)

        tar_values[m] = tar_values[m] - tar_values[m].min()
        if tar_values[m].max() != 0:
            tar_values[m] = tar_values[m] / tar_values[m].max()
        tar_values[m] = tar_values[m] * 254

        # Project from sensor_gravity into camera
        uv = (intrins[cam] @ H_camera__sensor_gravity[:3, :] @ points.T).T
        mask_in_front_of_camera = uv[:, 2] > 0
        uv = uv[mask_in_front_of_camera, :2] / uv[mask_in_front_of_camera, 2, None]
        uv = uv.type(torch.long)

        H, W = imgs[cam].shape[1], imgs[cam].shape[2]

        mask_valid_height = ~torch.isnan(tar_values[mask_in_front_of_camera])[:, 0]

        mask_inside_frustrum = (
            (uv[:, 0] >= 0) * (uv[:, 1] >= 0) * (uv[:, 0] < W) * (uv[:, 1] < H)
        )
        m = mask_inside_frustrum * mask_valid_height
        tar_values = tar_values[m]
        uv = uv[m]

        # Prepare foreground and background image
        background = denormalize_img(imgs[cam])
        background.convert("RGBA")
        foreground = np.array(background).copy()
        foreground[:, :, :] = 0
        foreground = Image.fromarray(foreground).convert("RGBA")
        foreground.putalpha(0)
        draw = ImageDraw.Draw(foreground)
        w = 3
        uv = uv.cpu().tolist()
        for i in range(len(uv)):
            fill = (self._c_maps["RdYlBu"][255 - int(tar_values[i])]).tolist() + [130]
            draw.arc(
                [(uv[i][0], uv[i][1]), (uv[i][0] + w, uv[i][1] + w)],
                start=0,
                end=360,
                fill=tuple(fill),
                width=w,
            )

        background.paste(foreground, (0, 0), foreground)
        return np.array(background)

    @image_functionality
    def plot_dashboard_new(self, dashboard_log, **kwargs):
        dashboard = dashboard_log["plot_all_maps"]

        def scale_and_cat(img, dashboard, add_pre=True):
            factor = dashboard.shape[1] / img.shape[1]
            scaled_img = cv2.resize(
                img,
                dsize=(dashboard.shape[1], int(img.shape[0] * factor)),
                interpolation=cv2.INTER_CUBIC,
            )
            if add_pre:
                return np.concatenate([scaled_img, dashboard], axis=0)
            else:
                return np.concatenate([dashboard, scaled_img], axis=0)

        if "plot_raw_images" in dashboard_log.keys():
            dashboard = scale_and_cat(dashboard_log["plot_raw_images"], dashboard)
        if "project_pred_BEV_on_image" in dashboard_log.keys():
            dashboard = scale_and_cat(
                dashboard_log["project_pred_BEV_on_image"], dashboard
            )
        if "project_gt_BEV_on_image" in dashboard_log.keys():
            dashboard = scale_and_cat(
                dashboard_log["project_gt_BEV_on_image"], dashboard
            )

        return dashboard

    @image_functionality
    def project_BEV_on_image(
        self,
        imgs: torch.tensor,
        rots: torch.tensor,
        trans: torch.tensor,
        post_rots: torch.tensor,
        post_trans: torch.tensor,
        intrins: torch.tensor,
        target: torch.tensor,
        elevation: torch.tensor,
        grid_map_resolution: float = 0.5,
        cam: int = 0,
        max_cells_radius: int = 150,
        v_min=0,
        v_max=1,
        expand=False,
        **kwargs,
    ):
        """
        post_rots, post_trans are currently not used.
        """
        if type(cam) is list:
            out_imgs = []
            for c in cam:
                kwargs["not_log"] = True
                kwargs["store"] = False
                out_imgs.append(
                    self.project_BEV_on_image(
                        imgs=imgs,
                        rots=rots,
                        trans=trans,
                        post_rots=post_rots,
                        post_trans=post_trans,
                        intrins=intrins,
                        target=target,
                        elevation=elevation,
                        grid_map_resolution=grid_map_resolution,
                        cam=c,
                        v_min=v_min,
                        v_max=v_max,
                        max_cells_radius=max_cells_radius,
                        **kwargs,
                    )
                )
            return np.concatenate(out_imgs, axis=1)

        device = imgs.device
        margin = int((target.shape[1] / 2) - ((2**0.5) * max_cells_radius) - 1)

        target = target[:, margin:-(margin), margin:-(margin)]
        elevation = elevation[margin:-(margin), margin:-(margin)]

        if expand:
            import torch.nn.functional as F

            target[torch.isnan(target)] = -torch.inf
            target = F.max_pool2d(target, (7, 7), 1, padding=3)
            target[torch.isinf(target)] = 0
        # Create 3D points from a given gridmap
        N = int(target.shape[1] / 2)
        x_ = torch.arange(-N, N, 1, device=device) * grid_map_resolution
        y_ = torch.arange(-N, N, 1, device=device) * grid_map_resolution
        xv, yv = torch.meshgrid(x_, y_, indexing="ij")
        ones = torch.ones_like(xv)
        points = torch.stack([xv, yv, elevation, ones], dim=2).reshape(-1, 4)

        # Get transformation
        H_sensor_gravity__camera = torch.eye(4, device=device)
        H_sensor_gravity__camera[:3, :3] = rots[cam]
        H_sensor_gravity__camera[:3, 3] = trans[cam]
        H_camera__sensor_gravity = invert_se3(H_sensor_gravity__camera)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy()[:,:3])
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        # mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=17.0, origin=[0, 0, 0])
        # mesh_frame2.transform(H_sensor_gravity__camera.cpu().numpy())
        # o3d.visualization.draw_geometries([pcd, mesh_frame, mesh_frame2])

        tar_values = target.reshape(-1, 1)
        m = ~torch.isnan(tar_values)

        if v_max is not None:
            tar_values[m] = tar_values[m].clip(v_min, v_max)
            tar_values[m] -= v_min
            tar_values[m] /= v_max - v_min
        else:
            tar_values[m] = tar_values[m] - tar_values[m].min()
            if tar_values[m].max() != 0:
                tar_values[m] = tar_values[m] / tar_values[m].max()

        tar_values[m] = tar_values[m] * 255
        if not hasattr(self, "rectangles"):
            # Helper for drawing grid
            indices = torch.arange(0, tar_values.shape[0], 1, device=device).reshape(
                target.shape[1:]
            )
            rectangles = []
            rectangles_distance = []
            for i in range(0, indices.shape[0] - 1):
                for j in range(0, indices.shape[1] - 1):
                    rectangles.append(
                        [
                            indices[i, j],
                            indices[i + 1, j],
                            indices[i + 1, j + 1],
                            indices[i, j + 1],
                        ]
                    )
                    rectangles_distance.append(((i - N) ** 2 + (j - N) ** 2) ** 0.5)
            self.rectangles = torch.tensor(rectangles, device=device)
            self.rectangles_distance_mask = (
                torch.tensor(rectangles_distance, device=device) > max_cells_radius
            )

        # Project from sensor_gravity into camera
        # intrins_float = intrins[cam].float()
        H_camera__sensor_gravity = H_camera__sensor_gravity.float()
        intrins = intrins.float()
        points = points.float()

        uv = (intrins[cam] @ H_camera__sensor_gravity[:3, :] @ points.T).T
        uv[uv[:, 2] < 0.001, 2] = -1
        uv_original = uv[:, :2] / uv[:, 2, None]

        H, W = imgs[cam].shape[1], imgs[cam].shape[2]
        rec = self.rectangles.reshape(-1)
        polygons = uv_original[rec]
        pol = polygons.reshape(-1, 4, 2)
        self.rectangles = self.rectangles.reshape(-1, 4)
        elevation = points[:, 2][rec].reshape(-1, 4)

        mask_in_front_of_camera = torch.all((uv[rec][:, 2] > 0).reshape(-1, 4), dim=1)
        mask_valid_height = torch.all(~torch.isnan(elevation), dim=1)
        mask_inside_frustrum = torch.any(
            (pol[:, :, 0] >= 0)
            * (pol[:, :, 1] >= 0)
            * (pol[:, :, 0] < W)
            * (pol[:, :, 1] < H),
            dim=1,
        )

        m = (
            mask_inside_frustrum * mask_valid_height * mask_in_front_of_camera
        )  # * self.rectangles_distance_mask
        pol = pol[m]

        # Prepare foreground and background image
        background = denormalize_img(imgs[cam])
        background.convert("RGBA")
        if m.sum() < 10:
            return background

        foreground = np.array(background).copy()
        foreground[:, :, :] = 0
        foreground = Image.fromarray(foreground).convert("RGBA")
        foreground.putalpha(0)

        # Convert target value into color
        tar_values = tar_values[self.rectangles[m, 0]]
        m = ~torch.isnan(tar_values)
        # TODO verify this is good
        tar_values[torch.isnan(tar_values)] = 0
        tar_values = tar_values.clip(0, 255).type(torch.long).clip(0, 255)

        H_img, W_img = imgs.shape[2], imgs.shape[3]
        if m.sum() == 0:
            print("Error plotting BEV ! Returning empty image.")
            return np.zeros((H_img, W_img, 3), dtype=np.uint8)

        fast = True
        if fast:
            try:
                indexing = (255 - tar_values).clip(0, 255)
                colors = torch.tensor(self._c_maps["RdYlBu"]).to(tar_values.device)[
                    indexing
                ][:, 0]
                colors = torch.cat(
                    [
                        colors,
                        torch.full((colors.shape[0], 1), 100, device=colors.device),
                    ],
                    dim=1,
                )
                colors = colors.type(torch.uint8)
                BS, C, H, W = imgs.shape
                foreground = self.draw_polygon_helper.draw_polygons(pol, colors)
                foreground = Image.fromarray(foreground.permute(1, 2, 0).cpu().numpy())
            except Exception as e:
                print(f"Error during poltting BEV ! Returning empty image. {e}")
                return np.zeros((H_img, W_img, 3), dtype=np.uint8)
        else:
            foreground_risk = foreground.copy()
            draw = ImageDraw.Draw(foreground)
            draw_risk = ImageDraw.Draw(foreground_risk)
            # Slow version but looks nicer

            fill = np.concatenate(
                [
                    self._c_maps["RdYlBu"][
                        (255 - tar_values).type(torch.long).clip(0, 255).cpu()
                    ][:, 0],
                    np.full((tar_values.shape[0], 1), 100),
                ],
                axis=1,
            )
            with Timer(f"loop over polygons, m sum {m.sum()}"):
                # Iterate over all polygons and draw on image
                pol = pol.tolist()
                for i in torch.where(m[:, 0])[0]:
                    xy = [tuple(xy) for xy in pol[i]]
                    draw.polygon(
                        xy, fill=tuple(fill[i]), outline=(0, 0, 0, 130), width=1
                    )
                for i in torch.where(~m[:, 0])[0]:
                    xy = [tuple(xy) for xy in pol[i]]
                    draw.polygon(
                        xy, fill=(1, 1, 1, 255), outline=(0, 0, 0, 130), width=1
                    )

                    # draw.polygon(xy, fill=(1, 1, 1, 255), outline=(0, 0, 0, 130), width=1)
                    # else:
                    # fill = (self._c_maps["RdYlBu"][255 - int(tar_values[i])]).tolist() + [100]
                    # pass
                    # if int(tar_values[i]) > 150:
                    #     draw_risk.polygon(xy, fill=tuple(fill), outline=(0, 0, 0, 130), width=1)

        background.paste(foreground, (0, 0), foreground)
        return np.array(background)

    @image_functionality
    def project_points_raw(
        self, imgs: torch.tensor, points_raw: torch.tensor, intrins, cam, **kwargs
    ):
        uv = (intrins[cam] @ points_raw["points"]).T
        mask_in_front_of_camera = uv[:, 2] > 0
        uv = uv[:, :2] / uv[:, 2, None]
        uv = uv.type(torch.long)
        H, W = imgs[cam].shape[1], imgs[cam].shape[2]

        mask_valid_height = ~torch.isnan(tar_values)[:, 0]

        mask_inside_frustrum = (
            (uv[:, 0] >= 0) * (uv[:, 1] >= 0) * (uv[:, 0] < W) * (uv[:, 1] < H)
        )
        m = mask_inside_frustrum * mask_valid_height * mask_in_front_of_camera
        tar_values = tar_values[m]
        uv = uv[m]

        # Prepare foreground and background image
        background = denormalize_img(imgs[cam])
        background.convert("RGBA")
        foreground = np.array(background).copy()
        foreground[:, :, :] = 0
        foreground = Image.fromarray(foreground).convert("RGBA")
        foreground.putalpha(0)
        draw = ImageDraw.Draw(foreground)
        w = 3
        uv = uv.cpu().tolist()
        for i in range(len(uv)):
            fill = (self._c_maps["RdYlBu"][255 - int(tar_values[i])]).tolist() + [130]
            draw.arc(
                [(uv[i][0], uv[i][1]), (uv[i][0] + w, uv[i][1] + w)],
                start=0,
                end=360,
                fill=tuple(fill),
                width=w,
            )
        background.paste(foreground, (0, 0), foreground)
        return np.array(background)

    @image_functionality
    def plot_detectron(
        self,
        img,
        seg,
        alpha=0.5,
        draw_bound=True,
        max_seg=40,
        colormap="Set2",
        overlay_mask=None,
        **kwargs,
    ):
        img = self.plot_image(denormalize_img(img), not_log=True)
        # assert seg.max() < max_seg and seg.min() >= 0, f"Seg out of Bounds: 0-{max_seg}, Given: {seg.min()}-{seg.max()}"
        try:
            np_seg = seg.clone().cpu().numpy()
        except Exception:
            pass
        np_seg = np_seg.astype(np.uint32)

        H, W, C = img.shape
        overlay = np.zeros_like(img)
        c_map = sns.color_palette(colormap, max_seg)

        uni = np.unique(np_seg)
        # Commented out center extraction code
        # centers = []
        for u in uni:
            m = np_seg == u
            try:
                col = np.uint8(np.array(c_map[u])[:3] * 255)
            except Exception as e:
                print(e)
                continue
            overlay[m] = col
            # segs_mask = skimage.measure.label(m)
            # regions = skimage.measure.regionprops(segs_mask)
            # regions.sort(key=lambda x: x.area, reverse=True)
            # cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]
            # centers.append((self._meta_data["stuff_classes"][u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        if overlay_mask is not None:
            try:
                overlay_mask = overlay_mask.cpu().numpy()
            except Exception:
                pass
            fore[overlay_mask] = 0

        img_new = Image.alpha_composite(
            Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore))
        )

        img_new = img_new.convert("RGB")
        mask = skimage.segmentation.mark_boundaries(
            np.array(img_new), np_seg, color=(255, 255, 255)
        )
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)

    @image_functionality
    def plot_detectron_cont(
        self, img, seg, alpha=0.3, max_val=1.0, colormap="RdYlBu", **kwargs
    ):
        img = self.plot_image(denormalize_img(img, not_log=True, store=False))
        assert (
            seg.max() <= max_val and seg.min() >= 0
        ), f"Seg out of Bounds: 0-{max_val}, Given: {seg.min()}-{seg.max()}"
        try:
            seg = seg.clone().cpu().numpy()
        except Exception:
            pass
        seg = np.uint8(seg.astype(np.float32) * 255)

        H, W, C = img.shape
        overlay = np.zeros_like(img)

        if colormap not in self._c_maps:
            self._c_maps[colormap] = np.array(
                [np.uint8(np.array(c) * 255) for c in sns.color_palette(colormap, 256)]
            )
        c_map = self._c_maps[colormap]

        uni = np.unique(seg)
        for u in uni:
            m = seg == u
            overlay[m] = c_map[u]

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(
            Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore))
        )
        img_new = np.array(img_new.convert("RGB"))
        return np.uint8(img_new)

    @image_functionality
    def plot_segmentation(self, seg, max_seg=40, colormap="Set2", **kwargs):
        try:
            seg = seg.clone().cpu().numpy()
        except Exception:
            pass

        if seg.shape[0] == 1:
            seg = seg[0]

        if seg.dtype == bool:
            max_seg = 2

        c_map = sns.color_palette(colormap, max_seg)

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)

        uni = np.unique(seg)

        for u in uni:
            img[seg == u] = np.uint8(np.array(c_map[u])[:3] * 255)

        return img

    @image_functionality
    def plot_image(self, img, **kwargs):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
            Range 0-1 or 0-255
        """
        try:
            img = img.clone().cpu().numpy()
        except Exception:
            pass

        if img.shape[2] == 3:
            pass
        elif img.shape[0] == 3:
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        else:
            raise Exception("Invalid Shape")
        if img.max() <= 1:
            img = img * 255

        img = np.uint8(img)
        return img


# if __name__ == "__main__":
#     from perception_bev_learning import BEV_ROOT_DIR
#     from perception_bev_learning.cfg import ExperimentParams

#     visu = LearningVisualizer(p_visu=os.path.join(BEV_ROOT_DIR, "results"), store=True)

#     params = ExperimentParams
#     from perception_bev_learning.dataset import BevDataset

#     bev_dataset = BevDataset(params.dataset_train)
#     sample = bev_dataset[0]

#     visu.plot_elevation_map(sample[-1][0], tag="elevation_map")


if __name__ == "__main__":
    from perception_bev_learning import BEV_ROOT_DIR
    from perception_bev_learning.cfg import ExperimentParams

    visu = LearningVisualizer(p_visu=os.path.join(BEV_ROOT_DIR, "results"), store=True)

    a = torch.zeros((512, 512))
    b = torch.arange(0, 1, 1 / 512)[:, None].repeat(1, 512)
    b.shape
    visu.plot_elevation_map_two(a, b, tag=f"test", cmap_name="turbo", v_max=12, v_min=0)

# if __name__ == "__main__":
#     from perception_bev_learning import BEV_ROOT_DIR
#     from perception_bev_learning.cfg import ExperimentParams

#     visu = LearningVisualizer(p_visu=os.path.join(BEV_ROOT_DIR, "results"), store=True)

#     params = ExperimentParams
#     from perception_bev_learning.dataset import BevDataset

#     # bev_dataset = BevDataset(params.dataset_train)
#     # sample = bev_dataset[0]

#     # aux = sample[7]
#     # elevation_raw = aux[2]
#     # res = visu.plot_elevation_map(elevation_raw.cpu().numpy())

#     # import pickle
#     # img = 20
#     # for i in range(100):
#     #     with open(f'/home/jonfrey/tmp_{i}.pickle', 'rb') as handle:
#     #         data = pickle.load(handle)
#     #         data["tag"]
#     #         if data["tag"].find("0041") != -1:
#     #             print(data["tag"], i)
#     #             img = i
#     import pickle

#     with open(f"/home/jonfrey/tmp.pickle", "rb") as handle:
#         data = pickle.load(handle)

#     import cv2
#     from PIL import Image

#     # denormalize_img( data['imgs'][0].cpu() ).show()

#     # data["target"] = data["current"][None]
#     # data["max_cells_radius"] = 100
#     res = visu.project_BEV_on_image(
#         imgs=data["imgs"],
#         rots=data["rots"],
#         trans=data["trans"],
#         post_rots=data["post_rots"],
#         post_trans=data["post_trans"],
#         intrins=data["intrins"],
#         target=data["pred"],
#         elevation=data["elevation"],
#         grid_map_resolution=data["grid_map_resolution"],
#         v_min=data["v_min"],
#         v_max=data["v_max"],
#         cam=data["cam"],
#         not_log=True,
#         store=False,
#         tag=f"nan",
#     )
#     Image.fromarray(res).show()

#     # with open(f'/home/jonfrey/tmp2_{img}.pickle', 'rb') as handle:
#     #     data2 = pickle.load(handle)

#     # img = visu.plot_all_maps(
#     #     elevation = data2["aux"][0],
#     #     **data2
#     # )
#     # Image.fromarray(img).show()
