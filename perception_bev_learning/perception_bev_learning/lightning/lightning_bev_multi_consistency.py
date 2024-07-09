from perception_bev_learning.visu import LearningVisualizer
from perception_bev_learning.utils import denormalize_img as d_img
from perception_bev_learning.loss import LossManagerMulti
from perception_bev_learning.utils import BevMeterMulti, Timer
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
from lightning import LightningModule
import torch.nn as nn
from torchvision.transforms.functional import center_crop
import torch.nn.functional as F

CMAP_TRAVERSABILITY = sns.color_palette("RdYlBu_r", as_cmap=True)
CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ERROR = sns.color_palette("vlag", as_cmap=True)
CMAP_TRAVERSABILITY.set_bad(color="black")
CMAP_ELEVATION.set_bad(color="black")
CMAP_ERROR.set_bad(color="black")

# Define the custom colormap
colors = [(0, 0, 0), paper_colors_rgb_f["cyan"]]
CMAP_LIDAR = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)


class LightningBEVMultiConsistency(LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = network
        self.store_tag = ""
        # TODO: Was this ever used on cluster
        # if cfg.trainer.strategy == "ddp":
        # self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)

        self._visu = LearningVisualizer(
            p_visu=self.hparams.visualizer.path,
            store=True,
            pl_model=self,
            log=True,
        )

        self._mode = "train"

        # Setup the list of ranges (Micro, Short)
        self.range_keys = [x for x in self.hparams.metrics.target_layers.keys()]

        self._meter = BevMeterMulti(self.hparams.metrics, self)  # TODO: verify

        self._loss_manager = LossManagerMulti(
            self.hparams.target_layers
        )  # TODO: verify

        self._visu_count = {"val": 0, "test": 0, "train": 0}

        self._cct = ClassTimer(
            objects=[self],
            names=["LightningBEV"],
            enabled=False,  # TODO (Pass this as a parameter)
        )

        self.scale_target = nn.ParameterDict()
        self.scale_aux = nn.ParameterDict()

        for gridmap_key in self.hparams.metrics.target_layers.keys():
            scale_target = torch.tensor(
                [
                    l.scale
                    for l in self.hparams.metrics.target_layers[
                        gridmap_key
                    ].layers.values()
                ]
            )[None, :, None, None]
            self.scale_target[gridmap_key] = nn.Parameter(
                scale_target, requires_grad=False
            )

            scale_aux = torch.tensor(
                [
                    l.scale
                    for l in self.hparams.metrics.aux_layers[
                        gridmap_key
                    ].layers.values()
                ]
            )[None, :, None, None]
            self.scale_aux[gridmap_key] = nn.Parameter(scale_aux, requires_grad=False)

        # TODO: Remove these hardcoded values and instead pass them as parameters
        # TODO: Check if they are used anywhere
        self.travmap_gt_key = "g_traversability_map_micro_gt"
        self.travmap_key = "g_traversability_map_micro"
        self.elemap_key = "g_raw_ele_map_micro"

        # Setting up the dictionary for layer name to Idx
        self.target_idxs_dict = self._meter.target_idxs_dict
        self.aux_idxs_dict = self._meter.aux_idxs_dict

        self.sequence_length = self.hparams.sequence_length
        self.prev_seq_key = ""

        self.aux_grd_key = "short" if "short" in self.range_keys else "micro"

    def unscale(self, pred, aux=False):
        pred_unscaled = {}

        if aux:
            s = self.scale_aux
        else:
            s = self.scale_target

        for gridmap_key in self.range_keys:
            pred_unscaled[gridmap_key] = pred[gridmap_key].clone() / s[gridmap_key]

        return pred_unscaled

    @accumulate_time
    def forward(self, batch: torch.tensor, seq_i: int, batch_idx: int):
        return self.net(
            imgs=batch["imgs"][:, seq_i, ...],
            rots=batch["rots"][:, seq_i, ...],
            trans=batch["trans"][:, seq_i, ...],
            intrins=batch["intrins"][:, seq_i, ...],
            post_rots=batch["post_rots"][:, seq_i, ...],
            post_trans=batch["post_trans"][:, seq_i, ...],
            pcd=batch["pcd"][seq_i],
            gvom=batch["gvom"][seq_i],
            aux=batch["aux"][self.aux_grd_key][:, seq_i, ...],
            H_sg_map=batch["H_sg_map"][:, seq_i, ...],
            batch_idx=batch_idx,
        )

    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visu.epoch = self.current_epoch

    @accumulate_time
    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        loss_total = 0
        loss_for_logging = {}
        
        ### Adding the consistency loss between the predictions
        loss_consistency_logging = torch.zeros(1).to(batch["target"]["micro"][0].device)
        
        for gridmap_key in self.range_keys:
            loss_for_logging[gridmap_key] = torch.zeros(
                batch["target"][gridmap_key][0].shape[1]
            ).to(batch["target"][gridmap_key][0].device)

        bptt_seq_length = batch["imgs"].shape[1]

        if batch_idx % self.sequence_length // bptt_seq_length == 0:
            self.net.reset_temporal(
                batch["aux"][self.aux_grd_key][:, 0, ...], batch["H_sg_map"][:, 0, ...]
            )

        self.net.detach_hidden_state()
        # print(f"batch_idx is {batch_idx}, idx is {batch['index']}")

        for i in range(bptt_seq_length):
            pred = self(batch, i, batch_idx)

            target = {}
            aux = {}

            for gridmap_key in self.range_keys:
                target[gridmap_key] = batch["target"][gridmap_key][:, i, ...]
                aux[gridmap_key] = batch["aux"][gridmap_key][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )
            
            ### Adding the consistency loss between the predictions
            cropped_short = center_crop(final_pred["short"][:, 1][:, None], (128, 128))
            downsampled_micro = F.interpolate(final_pred["micro"][:, 1][:, None], (128, 128), mode="bilinear", align_corners=True)
            loss_consistency = 0.1 * F.smooth_l1_loss(cropped_short, downsampled_micro, reduction="none").mean()
            
            loss_consistency_logging += loss_consistency.item()
            loss_total += loss_consistency
            
            for gridmap_key in self.range_keys:
                for i, layer in enumerate(
                    self.hparams.metrics.target_layers[gridmap_key].layers.keys()
                ):
                    loss_for_logging[gridmap_key][i] += loss[gridmap_key][i].item()
                    loss_total += torch.stack(loss[gridmap_key]).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="train",
                lm=self,
            )
            # self.visu(batch, final_pred, batch_idx, i)

        for gridmap_key in self.range_keys:
            for i, layer in enumerate(
                self.hparams.metrics.target_layers[gridmap_key].layers.keys()
            ):
                self.log(
                    f"train/loss/{gridmap_key}_{layer}",
                    loss_for_logging[gridmap_key][i].item() / bptt_seq_length,
                    prog_bar=True,
                    batch_size=self.hparams.batch_size,
                )

        self.log("train/loss/total", loss_total.item() / bptt_seq_length, prog_bar=True)
        
        ### Adding the consistency loss between the predictions
        self.log("train/loss/consistency", loss_consistency.item() / bptt_seq_length, prog_bar=True)
        
        return {
            "loss": loss_total,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @torch.no_grad()
    @accumulate_time
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
        # self._meter.update(
        #     target=self.unscale(outputs["target"]),
        #     pred=self.unscale(outputs["pred"]),
        #     aux=self.unscale(outputs["aux"], aux=True),
        #     mode="train",
        #     lm=self,
        # )

    @accumulate_time
    def on_train_epoch_end(self):
        if self.ct_enabled:
            print(self._cct)

        x = PrettyTable()
        x.field_names = ["Metric", "Value"]
        x.add_row(["Epochs", self.current_epoch])
        x.add_row(["Steps", self.global_step])

        for k, v in self.trainer.logged_metrics.items():
            if k.find("_step") == -1:
                x.add_row([k, round(v.item(), 4)])

        print(x)
        self._mode = "train"

    @accumulate_time
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    @accumulate_time
    def validation_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        loss_total = 0
        loss_for_logging = {}
        
        ### Adding the consistency loss between the predictions
        loss_consistency_logging = torch.zeros(1).to(batch["target"]["micro"][0].device)

        for gridmap_key in self.range_keys:
            loss_for_logging[gridmap_key] = torch.zeros(
                batch["target"][gridmap_key][0].shape[1]
            ).to(batch["target"][gridmap_key][0].device)

        bptt_seq_length = batch["imgs"].shape[1]

        if batch_idx % self.sequence_length // bptt_seq_length == 0:
            self.net.reset_temporal(
                batch["aux"][self.aux_grd_key][:, 0, ...], batch["H_sg_map"][:, 0, ...]
            )

        # print(f"batch_idx is {batch_idx}, idx is {batch['index']}")

        for i in range(bptt_seq_length):
            pred = self(batch, i, batch_idx)

            target = {}
            aux = {}

            for gridmap_key in self.range_keys:
                target[gridmap_key] = batch["target"][gridmap_key][:, i, ...]
                aux[gridmap_key] = batch["aux"][gridmap_key][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )
            
            ### Adding the consistency loss between the predictions
            cropped_short = center_crop(final_pred["short"][:, 1][:, None], (128, 128))
            downsampled_micro = F.interpolate(final_pred["micro"][:, 1][:, None], (128, 128), mode="bilinear", align_corners=True)
            loss_consistency = 0.1 * F.smooth_l1_loss(cropped_short, downsampled_micro, reduction="none").mean()
            
            loss_consistency_logging += loss_consistency.item()
            loss_total += loss_consistency

            for gridmap_key in self.range_keys:
                for i, layer in enumerate(
                    self.hparams.metrics.target_layers[gridmap_key].layers.keys()
                ):
                    loss_for_logging[gridmap_key][i] += loss[gridmap_key][i].item()
                    loss_total += torch.stack(loss[gridmap_key]).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="val",
                lm=self,
            )
            # self.visu(batch, final_pred, batch_idx, i)

        for gridmap_key in self.range_keys:
            for i, layer in enumerate(
                self.hparams.metrics.target_layers[gridmap_key].layers.keys()
            ):
                self.log(
                    f"val/loss/{gridmap_key}_{layer}",
                    loss_for_logging[gridmap_key][i].item() / bptt_seq_length,
                    prog_bar=True,
                    batch_size=self.hparams.batch_size,
                )

        self.log("val/loss/total", loss_total.item() / bptt_seq_length, prog_bar=True)
        ### Adding the consistency loss between the predictions
        self.log("val/loss/consistency", loss_consistency.item() / bptt_seq_length, prog_bar=True)
        
        return {
            "loss": loss_total,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @torch.no_grad()
    @accumulate_time
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self._meter.update(
            target=self.unscale(outputs["target"]),
            pred=self.unscale(outputs["pred"]),
            aux=self.unscale(outputs["aux"], aux=True),
            mode="val",
            lm=self,
        )

    @accumulate_time
    def on_validation_epoch_end(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    @accumulate_time
    def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        item_idx = batch["index"][:, 0]
        seq_key = self.trainer.test_dataloaders.dataset.get_sequence_key(
            item_idx[0].item(),
            self.hparams.visualizer.tag_with_front_camera,
        )

        loss_total = 0
        loss_for_logging = {}
        
        ### Adding the consistency loss between the predictions
        loss_consistency_logging = torch.zeros(1).to(batch["target"]["micro"][0].device)

        for gridmap_key in self.range_keys:
            loss_for_logging[gridmap_key] = torch.zeros(
                batch["target"][gridmap_key][0].shape[1]
            ).to(batch["target"][gridmap_key][0].device)

        bptt_seq_length = batch["imgs"].shape[1]

        if self.hparams.test_full_sequence:
            if str(seq_key) != str(self.prev_seq_key):
                print("Resetting hidden state")
                self.net.reset_temporal(
                    batch["aux"][self.aux_grd_key][:, 0, ...],
                    batch["H_sg_map"][:, 0, ...],
                )

        else:
            if batch_idx % self.sequence_length // bptt_seq_length == 0:
                self.net.reset_temporal(
                    batch["aux"][self.aux_grd_key][:, 0, ...],
                    batch["H_sg_map"][:, 0, ...],
                )

        for i in range(bptt_seq_length):
            pred = self(batch, i, batch_idx)

            target = {}
            aux = {}

            for gridmap_key in self.range_keys:
                target[gridmap_key] = batch["target"][gridmap_key][:, i, ...]
                aux[gridmap_key] = batch["aux"][gridmap_key][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )

            ### Adding the consistency loss between the predictions
            cropped_short = center_crop(final_pred["short"][:, 1][:, None], (128, 128))
            # upsampled_short = F.interpolate(cropped_short, (512, 512), mode="bilinear", align_corners=True)
            downsampled_micro = F.interpolate(final_pred["micro"][:, 1][:, None], (128, 128), mode="bilinear", align_corners=True)
            loss_consistency = 0.01 * F.smooth_l1_loss(cropped_short, downsampled_micro, reduction="none").mean()
            # micro = final_pred["micro"][:, 1][:, None]
            # loss_consistency = 0.01 * F.smooth_l1_loss(upsampled_short, micro, reduction="none").mean()
            
            loss_consistency_logging += loss_consistency.item()
            loss_total += loss_consistency
            
            for gridmap_key in self.range_keys:
                for i, layer in enumerate(
                    self.hparams.metrics.target_layers[gridmap_key].layers.keys()
                ):
                    loss_for_logging[gridmap_key][i] += loss[gridmap_key][i].item()
                    loss_total += torch.stack(loss[gridmap_key]).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="test",
                lm=self,
            )
            # self.visu(batch, final_pred, batch_idx, i)

        for gridmap_key in self.range_keys:
            for i, layer in enumerate(
                self.hparams.metrics.target_layers[gridmap_key].layers.keys()
            ):
                self.log(
                    f"test/loss/{gridmap_key}_{layer}",
                    loss_for_logging[gridmap_key][i].item() / bptt_seq_length,
                    prog_bar=True,
                    batch_size=self.hparams.batch_size,
                )

        self.log("test/loss/total", loss_total.item() / bptt_seq_length, prog_bar=True)
        ### Adding the consistency loss between the predictions
        self.log("test/loss/consistency", loss_consistency.item() / bptt_seq_length, prog_bar=True)
        
        return {
            "loss": loss_total,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @accumulate_time
    @torch.no_grad()
    def on_test_batch_end(self, outputs, batch, batch_idx):
        self._meter.update(
            target=self.unscale(outputs["target"]),
            pred=self.unscale(outputs["pred"]),
            aux=self.unscale(outputs["aux"], aux=True),
            mode="test",
            lm=self,
        )

    @accumulate_time
    def on_test_epoch_start(self):
        self._mode = "test"
        self._visu_count[self._mode] = 0
        # self.enable_dropout()

    def enable_dropout(self):
        for m in self.net.modules():
            if m.__class__.__name__.startswith("Dropout"):
                print(m)
                m.train()

    @accumulate_time
    def on_test_epoch_end(self):
        if self.ct_enabled:
            print(self._cct)

        # TODO: verify if this needed
        # self._metric.compute("test", self)

        x = PrettyTable()
        x.field_names = ["Metric", "Value"]
        x.add_row(["Epochs", self.current_epoch])
        x.add_row(["Steps", self.global_step])

        for k, v in self.trainer.logged_metrics.items():
            if k.find("_step") == -1:
                x.add_row([k, round(v.item(), 4)])
        print(x)

        file_path = join(
            self.hparams.path,
            f"{self.current_epoch}_{self._mode}_meter_results{self.store_tag}.pkl",
        )
        with open(file_path, "wb") as file:
            dic = {}
            for k, v in self.trainer.logged_metrics.items():
                dic[k] = v.item()

            pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            keys = np.unique(
                np.array(
                    [str(s)[:-10] for s in Path(self._visu._p_visu).rglob("*.png")]
                )
            ).tolist()
            keys = [k.split("/")[-1] for k in keys]
            for k in keys:
                # make sure to only create video for elements that are digits
                imgs = [
                    str(s)
                    for s in Path(self._visu._p_visu).rglob(f"{k}*.png")
                    if str(s)[-10:-4].isdigit()
                ]
                imgs.sort(key=lambda x: int(x[-10:-4]))
                if len(imgs) != 0:
                    clip = ImageSequenceClip(imgs, fps=3)
                    clip.write_videofile(
                        imgs[0].replace(".png", ".mp4").replace("_000000", "")
                    )
        except:
            print("Could not log video!")

    def get_scaled_layer(self, layer_key, layer_cfg, data, aux=False):
        if aux:
            idx_dict = self.aux_idxs_dict
        else:
            idx_dict = self.target_idxs_dict

        idx = idx_dict[layer_key]
        layer_out = data[:, idx].clone()
        layer_out /= layer_cfg[layer_key].scale
        return layer_out

    @torch.no_grad()
    @accumulate_time
    def visu(self, batch, pred, batch_idx, pred_std=None):
        with torch.no_grad():
            if not (
                getattr(self.hparams.visualizer, self._mode)
                > self._visu_count[self._mode]
                or getattr(self.hparams.visualizer, self._mode) == -1
            ):
                return

            if batch_idx % 10 == 0 and self.ct_enabled:
                print(self._cct)

            imgs = batch["imgs"]
            rots = batch["rots"]
            trans = batch["trans"]
            intrins = batch["intrins"]
            post_rots = batch["post_rots"]
            post_trans = batch["post_trans"]
            target = batch["target"]
            aux = batch["aux"]
            img_plots = batch["img_plots"]
            gmr = batch["gm_res"]
            pcd = batch["pcd"]
            gvom = batch["gvom"]
            item_idx = batch["index"]
            BS = imgs.shape[0]

            if (
                self.hparams.visualizer.plot_pcd_bev
                or self.hparams.visualizer.plot_all_maps
            ):
                pcd_data = voxelize_pcd_scans(
                    pcd["points"].clone().detach(),
                    pcd["batch"].clone().detach(),
                    pcd["scan"].clone().detach(),
                    (512, 512, 1),
                    (0.2, 0.2, 20),
                )

                gvom_data = voxelize_pcd_scans(
                    gvom["points"][:, :3].clone().detach(),
                    gvom["batch"].clone().detach(),
                    gvom["scan"].clone().detach(),
                    (512, 512, 1),
                    (0.2, 0.2, 20),
                )

            aux_layers_cfg = self.hparams.metrics.aux_layers
            current_elevation_raw = self.get_scaled_layer(
                "elevation_raw", aux_layers_cfg, aux, aux=True
            )
            current_elevation_raw_inpainted = self.get_scaled_layer(
                "elevation", aux_layers_cfg, aux, aux=True
            )
            target_elevation = self.get_scaled_layer(
                "elevation",
                self.hparams.metrics.target_layers,
                target,
            )
            pred_elevation = (
                pred[:, 1] / self.hparams.metrics.target_layers["elevation"].scale
            )

            # pred_elevation_std = (
            #     pred_std["elevation"]
            #     / self.hparams.metrics.target_layers["elevation"].scale
            # )
            # pred_wheel_risk_std = pred_std["wheel_risk"]

            for b in range(BS):
                if (
                    getattr(self.hparams.visualizer, self._mode)
                    > self._visu_count[self._mode]
                    or getattr(self.hparams.visualizer, self._mode) == -1
                ):
                    # img_idx_str = str(batch_idx * BS + b)
                    img_idx_str = str(item_idx[b].item())
                    img_idx_str = "0" * (6 - len(img_idx_str)) + img_idx_str

                    if self.hparams.visualizer.tag_by_dataloader:
                        if self._mode == "train":
                            seq_key = (
                                self.trainer.train_dataloader.dataset.get_sequence_key(
                                    item_idx[b].item(),
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        elif self._mode == "val":
                            seq_key = (
                                self.trainer.val_dataloaders.dataset.get_sequence_key(
                                    item_idx[b].item(),
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        elif self._mode == "test":
                            seq_key = (
                                self.trainer.test_dataloaders.dataset.get_sequence_key(
                                    item_idx[b].item(),
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        img_idx_str = seq_key + "_" + img_idx_str

                    if imgs is not None:
                        dashboard_log = {}

                        confidence_idx = self.aux_idxs_dict["confidence_gt"]
                        reliable_idx = self.aux_idxs_dict["reliable"]
                        m_confidence_gt = aux[b, confidence_idx] > 0.5
                        m_reliable = aux[b, reliable_idx] > 0.5
                        m_unobserved = m_confidence_gt * ~m_reliable

                        invalid_m = ~m_confidence_gt

                        inpainted_risk = aux[b, self.aux_idxs_dict["wheel_risk"]]
                        inpainted_risk[invalid_m] = torch.nan

                        inpainted_elevation = current_elevation_raw_inpainted[b]
                        inpainted_elevation[invalid_m] = torch.nan

                        if self.hparams.visualizer.plot_all_risks:
                            fatal_risk = self.hparams.metrics.fatal_risk
                            current_aux = self.aux_idxs_dict["wheel_risk"]
                            current_aux_non_inpainted = self.aux_idxs_dict[
                                "wheel_risk"
                            ]  # Switch to Non Inpainted

                            b_pred = (pred[b, 0] > fatal_risk).type(torch.float32)
                            b_aux_inp = (aux[b, current_aux] > fatal_risk).type(
                                torch.float32
                            )
                            b_aux_non_inp = (
                                aux[b, current_aux_non_inpainted] > fatal_risk
                            ).type(torch.float32)
                            b_target = (target[b, 0] > fatal_risk).type(torch.float32)

                            b_aux_inp[invalid_m] = torch.nan
                            b_aux_non_inp[invalid_m] = torch.nan
                            b_target[invalid_m] = torch.nan
                            error_wheel_risk = torch.abs(target[b, 0] - pred[b, 0])
                            maps = torch.stack(
                                [
                                    pred[b, 0],
                                    inpainted_risk,
                                    # pred_wheel_risk_std[b][0],
                                    error_wheel_risk,
                                    aux[b, current_aux_non_inpainted],
                                    target[b, 0],
                                    b_pred,
                                    # b_aux_inp,
                                    b_aux_non_inp,
                                    b_target,
                                    aux[b, confidence_idx],
                                ],
                                dim=0,
                            )

                            self._visu.plot_n_maps(
                                maps=maps[:, 6:-6, 6:-6],
                                titles=[
                                    "RoadRunner",
                                    "RACER-X-Inpainted",
                                    # "RoadRunner Std",
                                    "Risk - Error GT - RR",
                                    "RACER-X",
                                    "Ground Truth",
                                    "RoadRunner-B",
                                    # "RACER-X-Inpainted-B",
                                    "RACER-X-B",
                                    "Ground Truth-B",
                                    "Reliable-GT",
                                ],
                                color_maps=[CMAP_TRAVERSABILITY] * 2
                                + [CMAP_ELEVATION] * 1
                                + [CMAP_TRAVERSABILITY] * 5
                                + [CMAP_ELEVATION] * 1,
                                v_mins=[0.0] * 10,
                                v_maxs=[1.0] * 10,
                                tag=f"{self._mode}_traversability_layers_{img_idx_str}",
                            )
                            del (
                                maps,
                                b_pred,
                                b_aux_inp,
                                b_aux_non_inp,
                                b_target,
                                current_aux_non_inpainted,
                                current_aux,
                            )

                        if self.hparams.visualizer.plot_all_elevations:
                            error_elevation = torch.abs(
                                target_elevation[b] - pred_elevation[b]
                            )
                            maps = torch.stack(
                                [
                                    pred_elevation[b],
                                    # pred_elevation_std[b][0],
                                    error_elevation,
                                    current_elevation_raw[b],
                                    inpainted_elevation,
                                    target_elevation[b],
                                    aux[b, confidence_idx],
                                ],
                                dim=0,
                            )

                            self._visu.plot_n_maps(
                                maps=maps[:, 6:-6, 6:-6],
                                titles=[
                                    "RoadRunner",
                                    # "RoadRunner-std",
                                    "Error Map",
                                    "RACER-X-Raw",
                                    "RACER-X-Inpainted",
                                    "Ground Truth",
                                    "Reliable-GT",
                                ],
                                # color_maps=[CMAP_ELEVATION] * 1 + [CMAP_ERROR] * 2 + [CMAP_ELEVATION] * 4,
                                color_maps=[CMAP_ELEVATION] * 6,
                                v_mins=[-20] * 1 + [0] * 1 + [-20] * 3 + [0],
                                v_maxs=[20] * 1 + [4] * 1 + [20] * 3 + [1],
                                tag=f"{self._mode}_elevation_layers_{img_idx_str}",
                            )
                            del maps

                        if self.hparams.visualizer.plot_all_maps:
                            error_wheel_risk = torch.abs(target[b, 0] - pred[b, 0])
                            error_elevation = target_elevation[b] - pred_elevation[b]

                            maps = torch.stack(
                                [
                                    pred[b, 0],
                                    inpainted_risk,
                                    target[b, 0],
                                    error_wheel_risk,
                                    aux[b, confidence_idx],
                                    pred_elevation[b],
                                    inpainted_elevation,
                                    target_elevation[b],
                                    error_elevation,
                                    pcd_data[b, 0, 0],
                                    gvom_data[b, 0, 0],
                                ],
                                dim=0,
                            )
                            img = self._visu.plot_n_maps(
                                maps=maps[:, 6:-6, 6:-6],
                                titles=[
                                    "Risk - RoadRunner",
                                    "Risk - RACER-X",
                                    "Risk - Ground Truth",
                                    "Risk - Error GT - RR",
                                    "Reliable - Ground Truth",
                                    "Elev - RoadRunner",
                                    "Elev - RACER-X",
                                    "Elev - Ground Truth",
                                    "Elev - Error GT - RR",
                                    "Pointcloud",
                                    "GVOMcloud",
                                ],
                                color_maps=[CMAP_TRAVERSABILITY] * 3
                                + [CMAP_ELEVATION] * 6
                                + [CMAP_LIDAR] * 2,
                                v_mins=[0.0] * 5 + [-20] * 3 + [-3] * 1 + [0.0] * 2,
                                v_maxs=[1.0] * 5 + [20] * 3 + [3] * 1 + [1.0] * 2,
                                tag=f"{self._mode}_all_BEV_{img_idx_str}",
                                not_log=self.hparams.visualizer.plot_dashboard,
                                store=not self.hparams.visualizer.plot_dashboard,
                                store_svg=False,
                            )
                            dashboard_log["plot_all_maps"] = img
                            del maps

                        v_min = 0.0
                        v_max = 1.0
                        if self.hparams.visualizer.project_pcd_on_image:
                            self._visu.project_pcd_on_image(
                                imgs=img_plots[b].clone(),
                                rots=rots[b],
                                trans=trans[b],
                                intrins=intrins[b],
                                pcd_data=pcd_data[b, -1, 0],
                                grid_map_resolution=gmr[b],
                                cam=0,
                                tag=f"{self._mode}_project_pointcloud_on_image0_{img_idx_str}",
                            )

                        torch.cuda.empty_cache()
                        if self.hparams.visualizer.project_gt_BEV_on_image:
                            img = self._visu.project_BEV_on_image(
                                imgs=img_plots[b].clone(),
                                rots=rots[b],
                                trans=trans[b],
                                post_rots=post_rots[b],
                                post_trans=post_trans[b],
                                intrins=intrins[b],
                                target=target[b, 0][None],
                                elevation=target_elevation[b],
                                grid_map_resolution=gmr[b],
                                v_min=v_min,
                                v_max=v_max,
                                cam=list(self.hparams.visualizer.project_cams),
                                not_log=self.hparams.visualizer.plot_dashboard,
                                store=True,
                                tag=f"{self._mode}_project_gt_BEV_on_image0_{img_idx_str}",
                            )

                            dashboard_log["project_gt_BEV_on_image"] = img

                        if self.hparams.visualizer.project_pred_BEV_on_image:
                            img = self._visu.project_BEV_on_image(
                                imgs=img_plots[b].clone(),
                                rots=rots[b],
                                trans=trans[b],
                                post_rots=post_rots[b],
                                post_trans=post_trans[b],
                                intrins=intrins[b],
                                target=pred[b, 0][None],
                                elevation=pred_elevation[b],
                                grid_map_resolution=gmr[b],
                                v_min=v_min,
                                v_max=v_max,
                                cam=list(self.hparams.visualizer.project_cams),
                                not_log=self.hparams.visualizer.plot_dashboard,
                                store=True,
                                tag=f"{self._mode}_project_pred_BEV_on_image0_{img_idx_str}",
                            )
                            dashboard_log["project_pred_BEV_on_image"] = img

                        if self.hparams.visualizer.plot_raw_images:
                            all_imgs = self._visu.plot_list(
                                [
                                    np.array(d_img(img_plots[b, c].detach()))
                                    for c in self.hparams.visualizer.project_cams
                                ],
                                tag=f"{self._mode}_imgs_{img_idx_str}",
                                not_log=self.hparams.visualizer.plot_dashboard,
                                store=False,
                            )
                            dashboard_log["plot_raw_images"] = all_imgs

                        if self.hparams.visualizer.plot_dashboard:
                            self._visu.plot_dashboard_new(
                                dashboard_log=dashboard_log,
                                mode=self._mode,
                                tag=f"{self._mode}_dashboard_{img_idx_str}",
                            )

                        if self.hparams.visualizer.plot_pcd_bev:
                            self._visu.plot_pcd_bev(
                                maps=pcd_data.clone().detach()[b, :, 0],
                                tag=f"{self._mode}_pcd_bev{img_idx_str}",
                            )

                    self._visu_count[self._mode] += 1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # (TODO) Fix this at a later stage
        # if self._cfg.model.freeze_image_backbone:
        #     # Set requires grad to false at first
        #     try:
        #         for child in self._model.image_backbone.camencode.trunk.children():
        #             for param in child.parameters():
        #                 param.requires_grad = False
        #     except Exception as e:
        #         print("Failed to freeze EfficientNet Backbone")
        #     # Only optimize parameters that are not part of the EfficientNet backbone
        #     parameters = [
        #         v for k, v in self._model.named_parameters() if k.find("trunk") == -1
        #     ]
        # else:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}
