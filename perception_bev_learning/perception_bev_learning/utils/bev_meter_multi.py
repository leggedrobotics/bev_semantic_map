from torchmetrics import MeanSquaredError, MeanMetric
from perception_bev_learning.utils import (
    WeightedMeanSquaredError,
    WeightedMeanAbsoluteError,
    ValidMeanSquaredError,
    ValidMeanAbsoluteError,
    MaskedMeanAbsoluteError,
    MaskedMeanSquaredError,
    ValidMeanMetric,
    CellStatistics,
    f_ae,
    f_se,
)
from torch import nn
import torch
import numpy as np
from torchmetrics import Precision, Recall, F1Score, AUROC
from dataclasses import asdict
from pytictac import Timer
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from torchvision.transforms.functional import center_crop
import torch.nn.functional as F

CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ELEVATION.set_bad(color="black")


def crop(t, pad=6):
    # TODO: Padding for the short range should be 3 (and not default 6)
    if len(t.shape) == 2:
        return t[pad:-pad, pad:-pad]
    if len(t.shape) == 3:
        return t[:, pad:-pad, pad:-pad]
    if len(t.shape) == 4:
        return t[:, :, pad:-pad, pad:-pad]


def create_distance_mask(height, width, r):
    # Create grid of coordinates
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_x, center_y = width // 2, height // 2

    # Calculate distance of each pixel from the center
    distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create mask based on distance threshold
    mask = distances <= r
    mask = mask.unsqueeze(0)

    y_half = height // 2
    upper_mask = y < y_half
    upper_mask = upper_mask.unsqueeze(0)

    lower_mask = y >= y_half
    lower_mask = lower_mask.unsqueeze(0)

    mask_front = mask * lower_mask
    mask_back = mask * upper_mask

    return mask, mask_back, mask_front
    # return torch.tensor(mask, requires_grad=False), torch.tensor(mask_back, requires_grad=False), torch.tensor(mask_front, requires_grad=False)


class BevMeterMulti(nn.Module):
    def __init__(self, meter_cfg, lm, cfg=None):
        super().__init__()
        self.meter_cfg = meter_cfg
        self.key = "bm_"

        self.target_idxs_dict = {}
        self.aux_idxs_dict = {}

        self.ranges = {}
        self.ranges["micro"] = [25, 50, 75]
        self.ranges["short"] = [25, 50, 75, 100, 150]

        self.dims = {}
        self.dims["micro"] = (500, 500)
        self.dims["short"] = (244, 244)

        self.gmr_resolution = (
            {}
        )  # TODO: Pass this in the update step instead of hardcoding
        self.gmr_resolution["micro"] = 0.2
        self.gmr_resolution["short"] = 0.8

        self.range_mask = nn.ParameterDict()

        # Create the range metrics Masks
        for gridmap_key in self.meter_cfg.target_layers.keys():
            self.range_mask[gridmap_key] = nn.ParameterDict()
            for range in self.ranges[gridmap_key]:
                # Create a dict with the tensors for total, front and back
                self.range_mask[gridmap_key][f"{range}"] = nn.ParameterDict()
                range_pixels = range / self.gmr_resolution[gridmap_key]
                total, back, front = create_distance_mask(
                    self.dims[gridmap_key][0], self.dims[gridmap_key][1], range_pixels
                )
                self.range_mask[gridmap_key][f"{range}"]["total"] = nn.Parameter(
                    total, requires_grad=False
                )
                self.range_mask[gridmap_key][f"{range}"]["front"] = nn.Parameter(
                    front, requires_grad=False
                )
                self.range_mask[gridmap_key][f"{range}"]["back"] = nn.Parameter(
                    back, requires_grad=False
                )

        # Setting up the dictionary for layer name to Idx
        for key_gridmap in self.meter_cfg.target_layers.keys():
            self.target_idxs_dict[key_gridmap] = {
                l: j
                for j, l in enumerate(
                    meter_cfg.target_layers[key_gridmap].layers.keys()
                )
            }
            self.aux_idxs_dict[key_gridmap] = {
                l: j
                for j, l in enumerate(meter_cfg.aux_layers[key_gridmap].layers.keys())
            }

        if self.meter_cfg.hdd_statistic:
            print("HDD Statistic")
            lm.pred_target_recall = torch.nn.ModuleDict()
            lm.pred_target_precision = torch.nn.ModuleDict()
            lm.pred_target_f1 = torch.nn.ModuleDict()

            lm.current_target_recall = torch.nn.ModuleDict()
            lm.current_target_precision = torch.nn.ModuleDict()
            lm.current_target_f1 = torch.nn.ModuleDict()

        if self.meter_cfg.mse:
            lm.pred_target__mse = torch.nn.ModuleDict()
            lm.current_target__mse = torch.nn.ModuleDict()

        if self.meter_cfg.mae:
            lm.pred_target__mae = torch.nn.ModuleDict()
            lm.current_target__mae = torch.nn.ModuleDict()

        if self.meter_cfg.observed_vs_unobserved:
            lm.pred_target__observed_mse = torch.nn.ModuleDict()
            lm.pred_target__unobserved_mse = torch.nn.ModuleDict()

            # We will add the % Observed and % Unobserved and % DEM here as well
            lm.pred_target__observed_mae = torch.nn.ModuleDict()
            lm.pred_target__unobserved_mae = torch.nn.ModuleDict()
            lm.pred_target__dem_mae = torch.nn.ModuleDict()
            # For X-Racer observed vs unobserved
            # We will add the % Observed and Unobserved here as well
            lm.curr_target__observed_mae = torch.nn.ModuleDict()
            lm.curr_target__unobserved_mae = torch.nn.ModuleDict()
            lm.curr_target__dem_mae = torch.nn.ModuleDict()

        if self.meter_cfg.range_metrics:
            lm.range_metrics_mae = torch.nn.ModuleDict()

        if len(meter_cfg.target_layers.keys()) > 1:
            # We want to log the consistency loss as well
            lm.micro_short_consistency = torch.nn.ModuleDict()
            for m in meter_cfg.modes:
                lm.micro_short_consistency[
                    "micro_short_consistency_elevation_mae_" + m
                ] = ValidMeanAbsoluteError()

        for gridmap_key in meter_cfg.target_layers.keys():
            for lname, layer in meter_cfg.target_layers[gridmap_key].layers.items():
                for m in meter_cfg.modes:
                    if lname == "wheel_risk" or lname == "cost" or lname == "risk":
                        if self.meter_cfg.hdd_statistic:
                            lm.pred_target_precision[
                                gridmap_key + lname + "_" + m
                            ] = Precision(average="none", task="binary", num_classes=2)
                            lm.pred_target_recall[
                                gridmap_key + lname + "_" + m
                            ] = Recall(average="none", task="binary")
                            lm.pred_target_f1[gridmap_key + lname + "_" + m] = F1Score(
                                average="none", task="binary"
                            )

                            lm.current_target_precision[
                                gridmap_key + lname + "_" + m
                            ] = Precision(average="none", task="binary", num_classes=2)
                            lm.current_target_recall[
                                gridmap_key + lname + "_" + m
                            ] = Recall(average="none", task="binary")
                            lm.current_target_f1[
                                gridmap_key + lname + "_" + m
                            ] = F1Score(average="none", task="binary")

                    if self.meter_cfg.mse:
                        lm.pred_target__mse[
                            gridmap_key + lname + "_" + m
                        ] = MaskedMeanSquaredError()
                        # Lets add the % valid pixels
                        lm.pred_target__mse[
                            gridmap_key + lname + "_" + "valid_" + m
                        ] = ValidMeanMetric()
                        # Lets add the % valid pixels
                        lm.current_target__mse[
                            gridmap_key + lname + "_" + m
                        ] = ValidMeanSquaredError()
                        lm.current_target__mse[
                            gridmap_key + lname + "_" + "valid_" + m
                        ] = ValidMeanMetric()

                    if self.meter_cfg.mae:
                        lm.pred_target__mae[
                            gridmap_key + lname + "_" + m
                        ] = MaskedMeanAbsoluteError()
                        lm.current_target__mae[
                            gridmap_key + lname + "_" + m
                        ] = ValidMeanAbsoluteError()

                    if self.meter_cfg.observed_vs_unobserved:
                        if lname == "wheel_risk" or lname == "cost" or lname == "risk":
                            lm.pred_target__observed_mse[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanSquaredError()
                            lm.pred_target__unobserved_mse[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanSquaredError()
                        else:
                            lm.pred_target__observed_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.pred_target__observed_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()
                            lm.pred_target__unobserved_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.pred_target__unobserved_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()
                            lm.pred_target__dem_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.pred_target__dem_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()

                            # observed vs unobserved vs DEM for X-Racer
                            lm.curr_target__observed_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.curr_target__observed_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()
                            lm.curr_target__unobserved_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.curr_target__unobserved_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()
                            lm.curr_target__dem_mae[
                                gridmap_key + lname + "_" + m
                            ] = MaskedMeanAbsoluteError()
                            lm.curr_target__dem_mae[
                                gridmap_key + lname + "_valid_" + m
                            ] = ValidMeanMetric()

                    if self.meter_cfg.range_metrics:
                        if lname == "wheel_risk" or lname == "cost" or lname == "risk":
                            pass
                        else:
                            for range in self.ranges[gridmap_key]:
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_total_{range}_{m}"
                                ] = MaskedMeanAbsoluteError()
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_front_{range}_{m}"
                                ] = MaskedMeanAbsoluteError()
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_back_{range}_{m}"
                                ] = MaskedMeanAbsoluteError()

    @torch.no_grad()
    def update(self, target, pred, aux, mode, lm):
        m = mode
        if m not in self.meter_cfg.modes:
            return

        if len(self.meter_cfg.target_layers.keys()) > 1:
            # Log the consistency loss
            pred_micro = pred["micro"].detach()
            pred_short = pred["short"].detach()
            cropped_short = center_crop(pred_short[:, 1][:, None], (128, 128))
            downsampled_micro = F.interpolate(
                pred_micro[:, 1][:, None],
                (128, 128),
                mode="bilinear",
                align_corners=True,
            )
            lm.micro_short_consistency["micro_short_consistency_elevation_mae_" + m](
                cropped_short, downsampled_micro
            )
            lm.log(
                f"{m}/micro_short_consistency_elevation_mae",
                lm.micro_short_consistency[
                    "micro_short_consistency_elevation_mae_" + m
                ],
                on_epoch=True,
                on_step=False,
            )

        for gridmap_key in self.meter_cfg.target_layers.keys():
            # perform cropping
            target_c = crop(target[gridmap_key].detach())
            pred_c = crop(pred[gridmap_key].detach())
            aux_c = crop(aux[gridmap_key].detach())
            m_confidence_gt = (
                aux_c[:, self.aux_idxs_dict[gridmap_key]["confidence_gt"]] > 0.1
            )
            m_total = torch.ones_like(m_confidence_gt)
            m_reliable = aux_c[:, self.aux_idxs_dict[gridmap_key]["reliable"]] > 0.5
            m_unobserved = m_confidence_gt * ~m_reliable

            for j, (lname, layer) in enumerate(
                self.meter_cfg.target_layers[gridmap_key].layers.items()
            ):
                current = aux_c[:, self.aux_idxs_dict[gridmap_key][lname]]

                if lname == "wheel_risk" or lname == "cost" or lname == "risk":
                    if self.meter_cfg.hdd_statistic:
                        b_target = target_c[:, j] > self.meter_cfg.fatal_risk
                        b_current = current > self.meter_cfg.fatal_risk
                        b_pred = pred_c[:, j] > self.meter_cfg.fatal_risk
                        mask = ~torch.isnan(target_c[:, j]) * m_confidence_gt
                        lm.pred_target_precision[gridmap_key + lname + "_" + m](
                            b_pred[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/Pred/Precision_{lname}",
                            lm.pred_target_precision[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.pred_target_recall[gridmap_key + lname + "_" + m](
                            b_pred[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/Pred/Recall_{lname}",
                            lm.pred_target_recall[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.pred_target_f1[gridmap_key + lname + "_" + m](
                            b_pred[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/Pred/F1_{lname}",
                            lm.pred_target_f1[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.current_target_precision[gridmap_key + lname + "_" + m](
                            b_current[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/XRacer/Precision_{lname}",
                            lm.current_target_precision[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.current_target_recall[gridmap_key + lname + "_" + m](
                            b_current[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/XRacer/Recall_{lname}",
                            lm.current_target_recall[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.current_target_f1[gridmap_key + lname + "_" + m](
                            b_current[mask], b_target[mask]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/HDD/XRacer/F1_{lname}",
                            lm.current_target_f1[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.observed_vs_unobserved:
                        lm.pred_target__observed_mse[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_reliable
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Observed_MSE",
                            lm.pred_target__observed_mse[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.pred_target__unobserved_mse[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_unobserved
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Unobserved_MSE",
                            lm.pred_target__unobserved_mse[
                                gridmap_key + lname + "_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.mse:
                        lm.pred_target__mse[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Total_MSE",
                            lm.pred_target__mse[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.current_target__mse[gridmap_key + lname + "_" + m](
                            current, target_c[:, j]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Total_MSE",
                            lm.current_target__mse[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        # MSE Valid Wheel Risk predictions
                        lm.pred_target__mse[gridmap_key + lname + "_valid_" + m](
                            pred_c[:, j], target_c[:, j], m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred_valid_total",
                            lm.pred_target__mse[
                                gridmap_key + lname + "_" + "valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.current_target__mse[gridmap_key + lname + "_valid_" + m](
                            current, target_c[:, j], m_total
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer_valid_total",
                            lm.current_target__mse[
                                gridmap_key + lname + "_" + "valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.mae:
                        lm.pred_target__mae[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Total_MAE",
                            lm.pred_target__mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.current_target__mae[gridmap_key + lname + "_" + m](
                            current, target_c[:, j]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Total_MAE",
                            lm.current_target__mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                else:
                    if self.meter_cfg.observed_vs_unobserved:
                        lm.pred_target__observed_mae[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_reliable
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Observed_MAE",
                            lm.pred_target__observed_mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.pred_target__observed_mae[
                            gridmap_key + lname + "_valid_" + m
                        ](pred_c[:, j], target_c[:, j], m_reliable)
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred_valid_observed",
                            lm.pred_target__observed_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.pred_target__unobserved_mae[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], m_unobserved
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Unobserved_MAE",
                            lm.pred_target__unobserved_mae[
                                gridmap_key + lname + "_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.pred_target__unobserved_mae[
                            gridmap_key + lname + "_valid_" + m
                        ](pred_c[:, j], target_c[:, j], m_unobserved)
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred_valid_unobserved",
                            lm.pred_target__unobserved_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.pred_target__dem_mae[gridmap_key + lname + "_" + m](
                            pred_c[:, j], target_c[:, j], ~m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/DEM_MAE",
                            lm.pred_target__dem_mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.pred_target__dem_mae[gridmap_key + lname + "_valid_" + m](
                            pred_c[:, j], target_c[:, j], ~m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred_valid_DEM",
                            lm.pred_target__dem_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                        # X-Racer Observed V.S Unobserved
                        lm.curr_target__observed_mae[gridmap_key + lname + "_" + m](
                            current, target_c[:, j], m_reliable
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Observed_MAE",
                            lm.curr_target__observed_mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.curr_target__observed_mae[
                            gridmap_key + lname + "_valid_" + m
                        ](current, target_c[:, j], m_reliable)
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer_valid_observed",
                            lm.curr_target__observed_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.curr_target__unobserved_mae[gridmap_key + lname + "_" + m](
                            current, target_c[:, j], m_unobserved
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Unobserved_MAE",
                            lm.curr_target__unobserved_mae[
                                gridmap_key + lname + "_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.curr_target__unobserved_mae[
                            gridmap_key + lname + "_valid_" + m
                        ](current, target_c[:, j], m_unobserved)
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer_valid_unobserved",
                            lm.curr_target__unobserved_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                        lm.curr_target__dem_mae[gridmap_key + lname + "_" + m](
                            current, target_c[:, j], ~m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/DEM_MAE",
                            lm.curr_target__dem_mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.curr_target__dem_mae[gridmap_key + lname + "_valid_" + m](
                            current, target_c[:, j], ~m_confidence_gt
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer_valid_DEM",
                            lm.curr_target__dem_mae[
                                gridmap_key + lname + "_valid_" + m
                            ],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.mse:
                        lm.pred_target__mse[gridmap_key + lname + "_" + m](
                            pred_c[:, j],
                            target_c[:, j],
                            torch.ones_like(m_confidence_gt),
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Total_MSE",
                            lm.pred_target__mse[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.current_target__mse[gridmap_key + lname + "_" + m](
                            current, target_c[:, j]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Total_MSE",
                            lm.current_target__mse[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.mae:
                        lm.pred_target__mae[gridmap_key + lname + "_" + m](
                            pred_c[:, j],
                            target_c[:, j],
                            torch.ones_like(m_confidence_gt),
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/Pred/Total_MAE",
                            lm.pred_target__mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )
                        lm.current_target__mae[gridmap_key + lname + "_" + m](
                            current, target_c[:, j]
                        )
                        lm.log(
                            f"{m}/{gridmap_key}/{lname}/XRacer/Total_MAE",
                            lm.current_target__mae[gridmap_key + lname + "_" + m],
                            on_epoch=True,
                            on_step=False,
                        )

                    if self.meter_cfg.range_metrics:
                        for range in self.ranges[gridmap_key]:
                            B = pred_c.shape[0]
                            range_mask = self.range_mask[gridmap_key][f"{range}"][
                                "total"
                            ].repeat(B, 1, 1)
                            mask_front = self.range_mask[gridmap_key][f"{range}"][
                                "front"
                            ].repeat(B, 1, 1)
                            mask_back = self.range_mask[gridmap_key][f"{range}"][
                                "back"
                            ].repeat(B, 1, 1)

                            lm.range_metrics_mae[
                                f"{gridmap_key}_{lname}_total_{range}_{m}"
                            ](
                                pred_c[:, j],
                                target_c[:, j],
                                range_mask,
                            )
                            lm.log(
                                f"{m}/{gridmap_key}/{lname}/Range/Total_{range}",
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_total_{range}_{m}"
                                ],
                                on_epoch=True,
                                on_step=False,
                            )

                            lm.range_metrics_mae[
                                f"{gridmap_key}_{lname}_front_{range}_{m}"
                            ](
                                pred_c[:, j],
                                target_c[:, j],
                                mask_front,
                            )
                            lm.log(
                                f"{m}/{gridmap_key}/{lname}/Range/Front_{range}",
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_front_{range}_{m}"
                                ],
                                on_epoch=True,
                                on_step=False,
                            )

                            lm.range_metrics_mae[
                                f"{gridmap_key}_{lname}_back_{range}_{m}"
                            ](
                                pred_c[:, j],
                                target_c[:, j],
                                mask_back,
                            )
                            lm.log(
                                f"{m}/{gridmap_key}/{lname}/Range/Back_{range}",
                                lm.range_metrics_mae[
                                    f"{gridmap_key}_{lname}_back_{range}_{m}"
                                ],
                                on_epoch=True,
                                on_step=False,
                            )
