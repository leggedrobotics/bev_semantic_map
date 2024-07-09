from torchmetrics import MeanSquaredError
from perception_bev_learning.utils import (
    WeightedMeanSquaredError,
    WeightedMeanAbsoluteError,
    ValidMeanSquaredError,
    ValidMeanAbsoluteError,
    MaskedMeanAbsoluteError,
    MaskedMeanSquaredError,
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


def crop(t, pad=6):
    if len(t.shape) == 2:
        return t[pad:-pad, pad:-pad]
    if len(t.shape) == 3:
        return t[:, pad:-pad, pad:-pad]
    if len(t.shape) == 4:
        return t[:, :, pad:-pad, pad:-pad]


class BevMeter(nn.Module):
    def __init__(self, meter_cfg, lm, cfg=None):
        super().__init__()
        self.meter_cfg = meter_cfg
        self.key = "bm_"
        self.aux_idxs_dict = {l: j for j, l in enumerate(meter_cfg.aux_layers.keys())}

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
            lm.pred_target__observed_mae = torch.nn.ModuleDict()
            lm.pred_target__unobserved_mae = torch.nn.ModuleDict()
            lm.pred_target__dem_mae = torch.nn.ModuleDict()

        for lname, layer in meter_cfg.target_layers.items():
            for m in meter_cfg.modes:
                if lname == "wheel_risk":
                    if self.meter_cfg.hdd_statistic:
                        lm.pred_target_precision[
                            self.key + lname + "_" + m
                        ] = Precision(average="none", task="binary", num_classes=2)
                        lm.pred_target_recall[self.key + lname + "_" + m] = Recall(
                            average="none", task="binary"
                        )
                        lm.pred_target_f1[self.key + lname + "_" + m] = F1Score(
                            average="none", task="binary"
                        )

                        lm.current_target_precision[
                            self.key + lname + "_" + m
                        ] = Precision(average="none", task="binary", num_classes=2)
                        lm.current_target_recall[self.key + lname + "_" + m] = Recall(
                            average="none", task="binary"
                        )
                        lm.current_target_f1[self.key + lname + "_" + m] = F1Score(
                            average="none", task="binary"
                        )

                if self.meter_cfg.mse:
                    lm.pred_target__mse[
                        self.key + lname + "_" + m
                    ] = MaskedMeanSquaredError()
                    lm.current_target__mse[
                        self.key + lname + "_" + m
                    ] = ValidMeanSquaredError()

                if self.meter_cfg.mae:
                    lm.pred_target__mae[
                        self.key + lname + "_" + m
                    ] = MaskedMeanAbsoluteError()
                    lm.current_target__mae[
                        self.key + lname + "_" + m
                    ] = ValidMeanAbsoluteError()

                if self.meter_cfg.observed_vs_unobserved:
                    if lname == "wheel_risk":
                        lm.pred_target__observed_mse[
                            self.key + lname + "_" + m
                        ] = MaskedMeanSquaredError()
                        lm.pred_target__unobserved_mse[
                            self.key + lname + "_" + m
                        ] = MaskedMeanSquaredError()
                    else:
                        lm.pred_target__observed_mae[
                            self.key + lname + "_" + m
                        ] = MaskedMeanAbsoluteError()
                        lm.pred_target__unobserved_mae[
                            self.key + lname + "_" + m
                        ] = MaskedMeanAbsoluteError()
                        lm.pred_target__dem_mae[
                            self.key + lname + "_" + m
                        ] = MaskedMeanAbsoluteError()

    @torch.no_grad()
    def update(self, target, pred, aux, mode, lm):
        m = mode
        if m not in self.meter_cfg.modes:
            return

        # perform cropping
        target = crop(target.detach())
        pred = crop(pred.detach())
        aux = crop(aux.detach())
        m_confidence_gt = aux[:, self.aux_idxs_dict["confidence_gt"]] > 0.5
        m_reliable = aux[:, self.aux_idxs_dict["reliable"]] > 0.5
        m_unobserved = m_confidence_gt * ~m_reliable

        for j, (lname, layer) in enumerate(self.meter_cfg.target_layers.items()):
            current = aux[:, layer.aux_id]

            if lname == "wheel_risk":
                if self.meter_cfg.hdd_statistic:
                    b_target = target[:, j] > self.meter_cfg.fatal_risk
                    b_current = current > self.meter_cfg.fatal_risk
                    b_pred = pred[:, j] > self.meter_cfg.fatal_risk
                    mask = ~torch.isnan(target[:, j]) * ~torch.isnan(
                        aux[:, layer.aux_id]
                    )
                    lm.pred_target_precision[self.key + lname + "_" + m](
                        b_pred[mask], b_target[mask]
                    )
                    lm.log(
                        "pred_target_precision_" + self.key + lname + "_" + m,
                        lm.pred_target_precision[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.pred_target_recall[self.key + lname + "_" + m](
                        b_pred[mask], b_target[mask]
                    )
                    lm.log(
                        "pred_target_recall_" + self.key + lname + "_" + m,
                        lm.pred_target_recall[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.pred_target_f1[self.key + lname + "_" + m](
                        b_pred[mask], b_target[mask]
                    )
                    lm.log(
                        "pred_target_f1_" + self.key + lname + "_" + m,
                        lm.pred_target_f1[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.current_target_precision[self.key + lname + "_" + m](
                        b_current[mask], b_target[mask]
                    )
                    lm.log(
                        "current_target_precision_" + self.key + lname + "_" + m,
                        lm.current_target_precision[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.current_target_recall[self.key + lname + "_" + m](
                        b_current[mask], b_target[mask]
                    )
                    lm.log(
                        "current_target_recall_" + self.key + lname + "_" + m,
                        lm.current_target_recall[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.current_target_f1[self.key + lname + "_" + m](
                        b_current[mask], b_target[mask]
                    )
                    lm.log(
                        "current_target_f1_" + self.key + lname + "_" + m,
                        lm.current_target_f1[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                if self.meter_cfg.observed_vs_unobserved:
                    lm.pred_target__observed_mse[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_reliable
                    )
                    lm.log(
                        "pred_target__observed_mse_" + self.key + lname + "_" + m,
                        lm.pred_target__observed_mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                    lm.pred_target__unobserved_mse[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_unobserved
                    )
                    lm.log(
                        "pred_target__unobserved_mse_" + self.key + lname + "_" + m,
                        lm.pred_target__unobserved_mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                if self.meter_cfg.mse:
                    lm.pred_target__mse[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_confidence_gt
                    )
                    lm.log(
                        "pred_target__mse_" + self.key + lname + "_" + m,
                        lm.pred_target__mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.current_target__mse[self.key + lname + "_" + m](
                        current, target[:, j]
                    )
                    lm.log(
                        "current_target__mse_" + self.key + lname + "_" + m,
                        lm.current_target__mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                if self.meter_cfg.mae:
                    lm.pred_target__mae[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_confidence_gt
                    )
                    lm.log(
                        "pred_target__mae_" + self.key + lname + "_" + m,
                        lm.pred_target__mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.current_target__mae[self.key + lname + "_" + m](
                        current, target[:, j]
                    )
                    lm.log(
                        "current_target__mae_" + self.key + lname + "_" + m,
                        lm.current_target__mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

            else:
                if self.meter_cfg.observed_vs_unobserved:
                    lm.pred_target__observed_mae[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_reliable
                    )
                    lm.log(
                        "pred_target__observed_mae_" + self.key + lname + "_" + m,
                        lm.pred_target__observed_mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.pred_target__unobserved_mae[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], m_unobserved
                    )
                    lm.log(
                        "pred_target__unobserved_mae_" + self.key + lname + "_" + m,
                        lm.pred_target__unobserved_mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.pred_target__dem_mae[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], ~m_confidence_gt
                    )
                    lm.log(
                        "pred_target__dem_mae_" + self.key + lname + "_" + m,
                        lm.pred_target__dem_mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                if self.meter_cfg.mse:
                    lm.pred_target__mse[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], torch.ones_like(m_confidence_gt)
                    )
                    lm.log(
                        "pred_target__mse_" + self.key + lname + "_" + m,
                        lm.pred_target__mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.current_target__mse[self.key + lname + "_" + m](
                        current, target[:, j]
                    )
                    lm.log(
                        "current_target__mse_" + self.key + lname + "_" + m,
                        lm.current_target__mse[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )

                if self.meter_cfg.mae:
                    lm.pred_target__mae[self.key + lname + "_" + m](
                        pred[:, j], target[:, j], torch.ones_like(m_confidence_gt)
                    )
                    lm.log(
                        "pred_target__mae_" + self.key + lname + "_" + m,
                        lm.pred_target__mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
                    lm.current_target__mae[self.key + lname + "_" + m](
                        current, target[:, j]
                    )
                    lm.log(
                        "current_target__mae_" + self.key + lname + "_" + m,
                        lm.current_target__mae[self.key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )
