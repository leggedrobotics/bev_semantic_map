from perception_bev_learning.visu import LearningVisualizer
from perception_bev_learning.utils import denormalize_img as d_img
from perception_bev_learning.loss import LossManager
from perception_bev_learning.utils import BevMeter, Timer
from perception_bev_learning.ops import voxelize_pcd_scans

import torch
import numpy as np
from os.path import join
from dataclasses import asdict
from prettytable import PrettyTable
from moviepy.editor import ImageSequenceClip
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from pytictac import ClassTimer, ClassContextTimer, accumulate_time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from perception_bev_learning.visu import paper_colors_rgb_f
from perception_bev_learning.lightning import LightningBEV
from lightning import LightningModule
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

CMAP_TRAVERSABILITY = sns.color_palette("RdYlBu_r", as_cmap=True)
CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ERROR = sns.color_palette("vlag", as_cmap=True)
CMAP_TRAVERSABILITY.set_bad(color="black")
CMAP_ELEVATION.set_bad(color="black")
CMAP_ERROR.set_bad(color="black")

# Define the custom colormap
colors = [(0, 0, 0), paper_colors_rgb_f["cyan"]]
CMAP_LIDAR = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)


class LightningBEVDepth(LightningBEV):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
    ) -> None:
        super().__init__(network, optimizer, scheduler, **kwargs)

        self.dbound = self.hparams.network.cfg_model.encoders.camera.vtransform.dbound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.downsample_factor = (
            self.hparams.network.cfg_model.encoders.camera.vtransform.feature_downsample
        )

    @accumulate_time
    def forward(self, batch: torch.tensor, batch_idx: int):
        return self.net(
            imgs=batch["imgs"],
            rots=batch["rots"],
            trans=batch["trans"],
            intrins=batch["intrins"],
            post_rots=batch["post_rots"],
            post_trans=batch["post_trans"],
            target_shape=batch["target"].shape,
            pcd=batch["pcd"],
            gvom=batch["gvom"],
            aux=batch["aux"],
        )

    @accumulate_time
    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        pred, depth_preds = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]
        depth_labels = batch["depths"]

        depth_loss = self.get_depth_loss(depth_labels, depth_preds)

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"train_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        self.log("train_depth_loss", depth_loss.item(), prog_bar=True)

        # loss = torch.stack(loss).sum() + depth_loss
        loss = torch.stack(loss).sum()

        self.visu(batch, final_pred, batch_idx)
        self.log("train_loss", loss.item(), prog_bar=True)

        return {
            "loss": loss,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        )
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction="none",
            ).sum() / max(1.0, fg_mask.sum())

        return 0.01 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape

        # Calculate the new height and width after cropping
        new_H = H - (H % self.downsample_factor)
        new_W = W - (W % self.downsample_factor)

        # Crop the input tensor to the new size
        gt_depths = gt_depths[:, :, :new_H, :new_W]

        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample_factor, W // self.downsample_factor
        )

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1
        ).view(-1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    @accumulate_time
    def validation_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        pred, depth_preds = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]
        depth_labels = batch["depths"]

        depth_loss = self.get_depth_loss(depth_labels, depth_preds)

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"val_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        self.log("val_depth_loss", depth_loss.item(), prog_bar=True)

        loss = torch.stack(loss).sum() + depth_loss

        self.visu(batch, final_pred, batch_idx)
        self.log("val_loss", loss.item(), prog_bar=True)

        return {
            "loss": loss,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @accumulate_time
    def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        pred, depth_preds = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]
        depth_labels = batch["depths"]

        depth_loss = self.get_depth_loss(depth_labels, depth_preds)

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"test_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        self.log("test_depth_loss", depth_loss.item(), prog_bar=True)

        loss = torch.stack(loss).sum() + depth_loss

        self.visu(batch, final_pred, batch_idx)
        self.log("test_loss", loss.item(), prog_bar=True)

        return {
            "loss": loss,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }
