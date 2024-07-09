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


class LightningBEVTemporalBatch(LightningBEV):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
    ) -> None:
        super().__init__(network, optimizer, scheduler, **kwargs)

        self.sequence_length = self.hparams.sequence_length

    @accumulate_time
    def forward(self, batch: torch.tensor, seq_i: int):
        (
            imgs,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            target,
            aux,
            *_,
            H_sg_map,
            pcd_new,
            gvom_new,
        ) = batch

        return self.net(
            imgs[:, seq_i, ...],
            rots[:, seq_i, ...],
            trans[:, seq_i, ...],
            intrins[:, seq_i, ...],
            post_rots[:, seq_i, ...],
            post_trans[:, seq_i, ...],
            target[:, seq_i, ...].shape,
            pcd_new[seq_i],
            gvom_new[seq_i],
            aux[:, seq_i, ...],
            H_sg_map[:, seq_i, ...],
        )

    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visu.epoch = self.current_epoch

    @accumulate_time
    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        loss_total = 0
        loss_for_logging = torch.zeros(batch[6][0].shape[1]).to(batch[6][0].device)

        # Pass a sample Aux for correct dimensions
        self.net.reset_temporal(batch[7][:, 0, ...])

        for i in range(self.sequence_length):
            pred = self(batch, i)
            target = batch[6][:, i, ...]
            aux = batch[7][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )

            for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
                loss_for_logging[i] += loss[i].item()

            loss_total += torch.stack(loss).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="train",
                lm=self,
            )
            # self.visu(batch, final_pred, batch_idx, i)
            loss = torch.stack(loss).sum()

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"train_{layer}",
                loss_for_logging[i].item() / self.sequence_length,
                prog_bar=True,
                batch_size=self.hparams.batch_size,
            )

        self.log("train_loss", loss_total.item() / self.sequence_length, prog_bar=True)

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
        self.log_epoch_statistic()

    @accumulate_time
    def on_validation_epoch_start(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0

    @accumulate_time
    def validation_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        loss_total = 0
        loss_for_logging = torch.zeros(batch[6][0].shape[1]).to(batch[6][0].device)

        self.net.reset_temporal(batch[7][:, 0, ...])
        for i in range(self.sequence_length):
            pred = self(batch, i)
            target = batch[6][:, i, ...]
            aux = batch[7][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )

            for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
                loss_for_logging[i] += loss[i].item()

            loss_total += torch.stack(loss).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="val",
                lm=self,
            )

            loss = torch.stack(loss).sum()

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"val_{layer}",
                loss_for_logging[i].item() / self.sequence_length,
                prog_bar=True,
                batch_size=self.hparams.batch_size,
            )

            # self.visu(batch, final_pred, batch_idx)
        self.log("val_loss", loss_total.item() / self.sequence_length, prog_bar=True)

        return {
            "loss": loss_total,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @torch.no_grad()
    @accumulate_time
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # self._meter.update(
        #     target=self.unscale(outputs["target"]),
        #     pred=self.unscale(outputs["pred"]),
        #     aux=self.unscale(outputs["aux"], aux=True),
        #     mode="val",
        #     lm=self,
        # )
        pass

    @accumulate_time
    def on_validation_epoch_end(self):
        self._mode = "val"
        self._visu_count[self._mode] = 0
        self.log_epoch_statistic()

    @accumulate_time
    def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        loss_total = 0
        loss_for_logging = torch.zeros(batch[6][0].shape[1]).to(batch[6][0].device)

        self.net.reset_temporal(batch[7][:, 0, ...])
        for i in range(self.sequence_length):
            pred = self(batch, i)
            target = batch[6][:, i, ...]
            aux = batch[7][:, i, ...]

            loss, final_pred = self._loss_manager.compute(
                pred, target, aux, self.aux_idxs_dict
            )

            for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
                loss_for_logging[i] += loss[i].item()

            loss_total += torch.stack(loss).sum()

            self._meter.update(
                target=self.unscale(target),
                pred=self.unscale(final_pred),
                aux=self.unscale(aux, aux=True),
                mode="test",
                lm=self,
            )

            loss = torch.stack(loss).sum()

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"test_{layer}",
                loss_for_logging[i].item() / self.sequence_length,
                prog_bar=True,
                batch_size=self.hparams.batch_size,
            )

            # self.visu(batch, final_pred, batch_idx)
        self.log("test_loss", loss_total.item() / self.sequence_length, prog_bar=True)

        return {
            "loss": loss_total,
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @accumulate_time
    @torch.no_grad()
    def on_test_batch_end(self, outputs, batch, batch_idx):
        # self._meter.update(
        #     target=self.unscale(outputs["target"]),
        #     pred=self.unscale(outputs["pred"]),
        #     aux=self.unscale(outputs["aux"], aux=True),
        #     mode="test",
        #     lm=self,
        # )
        pass

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

    @torch.no_grad()
    @accumulate_time
    def visu(self, batch, pred, batch_idx, seq_i=None, pred_std=None):
        with torch.no_grad():
            if not (
                getattr(self.hparams.visualizer, self._mode)
                > self._visu_count[self._mode]
                or getattr(self.hparams.visualizer, self._mode) == -1
            ):
                return

            if batch_idx % 10 == 0 and self.ct_enabled:
                print(self._cct)

            BS = batch[0].shape[0]
            (
                imgs,
                rots,
                trans,
                intrins,
                post_rots,
                post_trans,
                target,
                aux,
                img_plots,
                gmr,
                pcd_new,
                gvom_new,
            ) = batch

            if (
                self.hparams.visualizer.plot_pcd_bev
                or self.hparams.visualizer.plot_all_maps
            ):
                pcd_data = voxelize_pcd_scans(
                    pcd_new["points"].clone().detach(),
                    pcd_new["batch"].clone().detach(),
                    pcd_new["scan"].clone().detach(),
                    (512, 512, 1),
                    (0.2, 0.2, 20),
                )
                # print(f"GVOM new points shape {gvom_new['points'].shape}")
                gvom_data = voxelize_pcd_scans(
                    gvom_new["points"][:, :3].clone().detach(),
                    gvom_new["batch"].clone().detach(),
                    gvom_new["scan"].clone().detach(),
                    (512, 512, 1),
                    (0.2, 0.2, 20),
                )

            al = self.hparams.metrics.aux_layers
            current_elevation_raw = self.get_scaled_layer(
                "elevation_raw", self.elemap_key, al, aux
            )
            current_elevation_raw_inpainted = self.get_scaled_layer(
                "elevation", self.travmap_key, al, aux
            )
            target_elevation = self.get_scaled_layer(
                "elevation",
                self.travmap_gt_key,
                self.hparams.metrics.target_layers,
                target,
            )
            pred_elevation = (
                pred[:, 1] / self.hparams.metrics.target_layers["elevation"].scale
            )
            pred_elevation_std = (
                pred_std["elevation"]
                / self.hparams.metrics.target_layers["elevation"].scale
            )
            pred_wheel_risk_std = pred_std["wheel_risk"]

            for b in range(BS):
                if (
                    getattr(self.hparams.visualizer, self._mode)
                    > self._visu_count[self._mode]
                    or getattr(self.hparams.visualizer, self._mode) == -1
                ):
                    img_idx_str = str(batch_idx * BS + b)
                    img_idx_str = "0" * (6 - len(img_idx_str)) + img_idx_str

                    if self.hparams.visualizer.tag_by_dataloader:
                        if self._mode == "train":
                            seq_key = (
                                self.trainer.train_dataloader.dataset.get_sequence_key(
                                    batch_idx * BS + b,
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        elif self._mode == "val":
                            seq_key = (
                                self.trainer.val_dataloaders.dataset.get_sequence_key(
                                    batch_idx * BS + b,
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        elif self._mode == "test":
                            seq_key = (
                                self.trainer.test_dataloaders.dataset.get_sequence_key(
                                    batch_idx * BS + b,
                                    self.hparams.visualizer.tag_with_front_camera,
                                )
                            )
                        img_idx_str = seq_key + "_" + img_idx_str

                    if imgs is not None:
                        dashboard_log = {}

                        invalid_m = torch.bitwise_or(
                            aux[b, self.hparams.metrics.reliable_aux_layer] < 0.5,
                            torch.isnan(
                                aux[b, self.hparams.metrics.reliable_aux_layer]
                            ),
                        )
                        inpainted_risk = aux[
                            b, self.hparams.metrics.target_layers["wheel_risk"].aux_id
                        ]
                        inpainted_risk[invalid_m] = torch.nan

                        inpainted_elevation = current_elevation_raw_inpainted[b]
                        inpainted_elevation[invalid_m] = torch.nan

                        if self.hparams.visualizer.plot_all_risks:
                            fatal_risk = self.hparams.metrics.fatal_risk
                            current_aux = self.hparams.metrics.target_layers[
                                "wheel_risk"
                            ].aux_id
                            non_inpainted_name = list(
                                self.hparams.metrics.aux_layers.values()
                            )[current_aux].name.replace("_inpainted", "")
                            current_aux_non_inpainted = [
                                i
                                for i, l in enumerate(
                                    self.hparams.metrics.aux_layers.values()
                                )
                                if l.name == non_inpainted_name
                            ][0]

                            b_pred = (pred[b, 0] > fatal_risk).type(torch.float32)
                            b_aux_inp = (aux[b, current_aux] > fatal_risk).type(
                                torch.float32
                            )
                            b_aux_non_inp = (
                                aux[b, current_aux_non_inpainted] > fatal_risk
                            ).type(torch.float32)
                            b_target = (target[b, 0] > fatal_risk).type(torch.float32)

                            # b_pred[invalid_m ] = torch.nan
                            b_aux_inp[invalid_m] = torch.nan
                            b_aux_non_inp[invalid_m] = torch.nan
                            b_target[invalid_m] = torch.nan
                            error_wheel_risk = torch.abs(target[b, 0] - pred[b, 0])
                            maps = torch.stack(
                                [
                                    pred[b, 0],
                                    # inpainted_risk,
                                    pred_wheel_risk_std[b][0],
                                    error_wheel_risk,
                                    aux[b, current_aux_non_inpainted],
                                    target[b, 0],
                                    b_pred,
                                    # b_aux_inp,
                                    b_aux_non_inp,
                                    b_target,
                                    aux[b, self.hparams.metrics.reliable_aux_layer],
                                ],
                                dim=0,
                            )

                            self._visu.plot_n_maps(
                                maps=maps[:, 6:-6, 6:-6],
                                titles=[
                                    "RoadRunner",
                                    # "RACER-X-Inpainted",
                                    "RoadRunner Std",
                                    "Risk - Error GT - RR",
                                    "RACER-X",
                                    "Ground Truth",
                                    "RoadRunner-B",
                                    # "RACER-X-Inpainted-B",
                                    "RACER-X-B",
                                    "Ground Truth-B",
                                    "Reliable-GT",
                                ],
                                color_maps=[CMAP_TRAVERSABILITY] * 1
                                + [CMAP_ELEVATION] * 2
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
                                    pred_elevation_std[b][0],
                                    error_elevation,
                                    current_elevation_raw[b],
                                    inpainted_elevation,
                                    target_elevation[b],
                                    aux[b, self.hparams.metrics.reliable_aux_layer],
                                ],
                                dim=0,
                            )

                            self._visu.plot_n_maps(
                                maps=maps[:, 6:-6, 6:-6],
                                titles=[
                                    "RoadRunner",
                                    "RoadRunner-std",
                                    "Error Map",
                                    "RACER-X-Raw",
                                    "RACER-X-Inpainted",
                                    "Ground Truth",
                                    "Reliable-GT",
                                ],
                                # color_maps=[CMAP_ELEVATION] * 1 + [CMAP_ERROR] * 2 + [CMAP_ELEVATION] * 4,
                                color_maps=[CMAP_ELEVATION] * 7,
                                v_mins=[-20] * 1 + [0] * 2 + [-20] * 3 + [0],
                                v_maxs=[20] * 1 + [3] * 2 + [20] * 3 + [1],
                                tag=f"{self._mode}_elevation_layers_{img_idx_str}",
                            )
                            del maps

                        if self.hparams.visualizer.plot_all_maps:
                            error_wheel_risk = torch.abs(target[b, 0] - pred[b, 0])
                            error_elevation = torch.abs(
                                target_elevation[b] - pred_elevation[b]
                            )
                            maps = torch.stack(
                                [
                                    pred[b, 0],
                                    inpainted_risk,
                                    target[b, 0],
                                    error_wheel_risk,
                                    aux[b, self.hparams.metrics.reliable_aux_layer],
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
                                + [CMAP_ERROR] * 1
                                + [CMAP_ELEVATION] * 4
                                + [CMAP_ERROR] * 1
                                + [CMAP_LIDAR] * 2,
                                v_mins=[0.0] * 3
                                + [-1.0] * 1
                                + [0.0] * 1
                                + [-20] * 3
                                + [-3] * 1
                                + [0.0] * 2,
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
                                store=True,
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
