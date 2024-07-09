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


class LightningBEV(LightningModule):
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

        self._meter = BevMeter(self.hparams.metrics, self)

        self._loss_manager = LossManager(self.hparams.target_layers)  # TODO

        self._visu_count = {"val": 0, "test": 0, "train": 0}
        self._cct = ClassTimer(
            objects=[self],
            names=["LightningBEV"],
            enabled=False,  # TODO (Pass this as a parameter)
        )

        scale_target = torch.tensor(
            [l.scale for l in self.hparams.metrics.target_layers.values()]
        )[None, :, None, None]
        scale_aux = torch.tensor(
            [l.scale for l in self.hparams.metrics.aux_layers.values()]
        )[None, :, None, None]
        self.register_buffer("_scale_target", scale_target)
        self.register_buffer("_scale_aux", scale_aux)

        # TODO: Remove these hardcoded values and instead pass them as parameters
        self.travmap_gt_key = "g_traversability_map_micro_gt"
        self.travmap_key = "g_traversability_map_micro"
        self.elemap_key = "g_raw_ele_map_micro"

        # Setting up the dictionary for layer name to Idx
        self.target_idxs_dict = {
            l: j for j, l in enumerate(self.hparams.metrics.target_layers.keys())
        }
        self.aux_idxs_dict = {
            l: j for j, l in enumerate(self.hparams.metrics.aux_layers.keys())
        }

    def unscale(self, pred, aux=False):
        if aux:
            s = self._scale_aux
        else:
            s = self._scale_target
        return pred.clone() / s

    @accumulate_time
    def forward(self, batch: torch.tensor, batch_idx):
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
            batch_idx=batch_idx,
        )

    def on_train_epoch_start(self):
        self._mode = "train"
        self._visu_count[self._mode] = 0
        self._visu.epoch = self.current_epoch

    @accumulate_time
    def training_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )
        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"train_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        self.visu(batch, final_pred, batch_idx)
        self.log("train_loss", torch.stack(loss).sum(), prog_bar=True)

        return {
            "loss": torch.stack(loss).sum(),
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    @torch.no_grad()
    @accumulate_time
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._meter.update(
            target=self.unscale(outputs["target"]),
            pred=self.unscale(outputs["pred"]),
            aux=self.unscale(outputs["aux"], aux=True),
            mode="train",
            lm=self,
        )

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
        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]

        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )

        for i, layer in enumerate(self.hparams.metrics.target_layers.keys()):
            self.log(
                f"val_{layer}",
                loss[i].item(),
                prog_bar=True,
            )

        self.log("val_loss", torch.stack(loss).sum(), prog_bar=True)
        self.visu(batch, final_pred, batch_idx)

        return {
            "val_loss": torch.stack(loss).sum(),
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

    def log_epoch_statistic(self):
        # Get all results if you can
        m1 = [
            n
            for n in dir(self)
            if n.find("current_tar") != -1
            and type(getattr(self, n)) is torch.nn.ModuleDict
        ]
        m2 = [
            n
            for n in dir(self)
            if n.find("pred_tar") != -1
            and type(getattr(self, n)) is torch.nn.ModuleDict
        ]
        epoch_stats = {}

        for m in m1 + m2:
            module_dict = getattr(self, m)

            if m.find("cell_stat") != -1:
                for k, v in module_dict.items():
                    # Extract the hazzard statistics
                    if v.compute_hdd:
                        for k2 in ["precision_bins", "recall_bins", "f1_bins"]:
                            epoch_stats[m + "_" + k + "_" + k2] = [
                                metric.compute().tolist() for metric in getattr(v, k2)
                            ]

                    # Extract the maps mean
                    mean_map, var_map = v.get_maps()
                    epoch_stats[m + "_" + k + "_" + "mean_map"] = mean_map.cpu()
                    epoch_stats[m + "_" + k + "_" + "var_map"] = var_map.cpu()
                    for k2 in ["sum_squared_error", "sum_error", "total"]:
                        epoch_stats[m + "_" + k + "_" + k2] = getattr(v, k2).cpu()

                    # Extract the bins for the mean
                    mean_bins, var_bins = v.get_bins()
                    epoch_stats[m + "_" + k + "_" + "mean_bins"] = mean_bins.cpu()
                    epoch_stats[m + "_" + k + "_" + "var_bins"] = var_bins.cpu()
                    for k2 in [
                        "sum_squared_error_bins",
                        "sum_error_bins",
                        "total_bins",
                    ]:
                        epoch_stats[m + "_" + k + "_" + k2] = getattr(v, k2).cpu()

            else:
                for k, v in module_dict.items():
                    try:
                        epoch_stats[m + "_" + k] = v.compute().item()
                    except:
                        epoch_stats[m + "_" + k] = v.compute().tolist()

        file_name = join(
            self.hparams.path,
            f"{self.current_epoch}_{self._mode}_statistic_reverted_layer_scaling{self.store_tag}.pkl",
        )
        with open(file_name, "wb") as file:
            pickle.dump(epoch_stats, file, protocol=pickle.HIGHEST_PROTOCOL)

        # self._meter.reset_cell_statistics(self._mode, self)

    @accumulate_time
    def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
        pred = self(batch, batch_idx)
        target = batch["target"]
        aux = batch["aux"]
        loss, final_pred = self._loss_manager.compute(
            pred, target, aux, self.aux_idxs_dict
        )
        self.visu(batch, final_pred, batch_idx)
        self.log("test_loss", torch.stack(loss).sum(), prog_bar=True)

        return {
            "loss": torch.stack(loss).sum(),
            "pred": final_pred,
            "target": target,
            "aux": aux,
        }

    # @accumulate_time
    # """
    # For test time dropouts
    # """
    # def test_step(self, batch: any, batch_idx: int) -> torch.Tensor:
    #     if batch_idx < 2:
    #         self.enable_dropout()
    #     batch = batch
    #     pred = {}
    #     pred_std = {}
    #     pred_multiple = self(batch)
    #     pred_wheel_risk = pred_multiple["wheel_risk"]
    #     pred_elevation = pred_multiple["elevation"]
    #     predictions_ele = torch.stack(pred_elevation, dim=0)
    #     predictions_wr = torch.stack(pred_wheel_risk, dim=0)

    #     # Calculate the mean and standard deviation across the list
    #     mean_ele = torch.mean(predictions_ele, dim=0)
    #     std_ele = torch.std(predictions_ele, dim=0)

    #     mean_wr = torch.mean(predictions_wr, dim=0)
    #     std_wr = torch.std(predictions_wr, dim=0)

    #     pred["wheel_risk"] = mean_wr
    #     pred["elevation"] = mean_ele
    #     pred_std["wheel_risk"] = std_wr
    #     pred_std["elevation"] = std_ele

    #     target = batch[6]
    #     aux = batch[7]
    #     loss, final_pred = self._loss_manager.compute(pred, target)
    #     self.visu(batch, final_pred, batch_idx, pred_std)
    #     self.log("test_loss", torch.stack(loss).sum(), prog_bar=True)

    #     return {
    #         "loss": torch.stack(loss).sum(),
    #         "pred": final_pred,
    #         "target": target,
    #         "aux": aux,
    #     }

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

    # def get_scaled_layer(self, name, gridmap_topic, layer_cfg, data):
    #     layer_cfg = [v for v in layer_cfg.values()]
    #     idx = [
    #         j
    #         for j, l in enumerate(layer_cfg)
    #         if l.name == name and l.gridmap_topic == gridmap_topic
    #     ]
    #     if len(idx) != 1:
    #         return None

    #     idx = idx[0]
    #     layer_out = data[:, idx].clone()
    #     layer_out /= layer_cfg[idx].scale
    #     return layer_out

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
