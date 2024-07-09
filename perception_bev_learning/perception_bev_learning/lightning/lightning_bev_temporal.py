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

CMAP_TRAVERSABILITY = sns.color_palette("RdYlBu_r", as_cmap=True)
CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ERROR = sns.color_palette("vlag", as_cmap=True)
CMAP_TRAVERSABILITY.set_bad(color="black")
CMAP_ELEVATION.set_bad(color="black")
CMAP_ERROR.set_bad(color="black")

# Define the custom colormap
colors = [(0, 0, 0), paper_colors_rgb_f["cyan"]]
CMAP_LIDAR = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)


class LightningBEVTemporal(LightningBEV):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
    ) -> None:
        super().__init__(network, optimizer, scheduler, **kwargs)

        self.sequence_length = self.hparams.sequence_length
        self.automatic_optimization = True
        self.prev_seq_key = ""

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
            H_sg_map=batch["H_sg_map"],
            batch_idx=batch_idx,
        )

    @accumulate_time
    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        # TODO: Using sequence length and batch_idx, reset the hidden state
        if batch_idx % self.sequence_length == 0:
            self.net.reset_temporal(batch["aux"], batch["H_sg_map"])

        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]

        # print(f"Indices are {batch['index']}")

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )
        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"train_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        loss = torch.stack(loss).sum()
        self.visu(batch, final_pred, batch_idx)
        self.log("train_loss", loss.item(), prog_bar=True)

        return {
            "loss": loss,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @accumulate_time
    def validation_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        if batch_idx % self.sequence_length == 0:
            self.net.reset_temporal(batch["aux"], batch["H_sg_map"])

        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]

        # print(f"Indices are {batch['index']}")

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )
        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"val_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        loss = torch.stack(loss).sum()
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
        
        item_idx = batch["index"]
        seq_key = (self.trainer.test_dataloaders.dataset.get_sequence_key(item_idx[0].item(),
                                    self.hparams.visualizer.tag_with_front_camera,))

        if self.hparams.test_full_sequence:
            if (str(seq_key) != str(self.prev_seq_key)):

                print("Resetting hidden state")
                self.net.reset_temporal(batch["aux"], batch["H_sg_map"])
                
        else:
            if batch_idx % self.sequence_length == 0:
                self.net.reset_temporal(batch["aux"], batch["H_sg_map"])

        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]

        # print(f"Indices are {batch['index']}")

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )
        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"test_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        loss = torch.stack(loss).sum()
        self.visu(batch, final_pred, batch_idx)
        self.log("test_loss", loss.item(), prog_bar=True)

        self.prev_seq_key = seq_key

        return {
            "loss": loss,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }
