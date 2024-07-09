from torchmetrics import MeanSquaredError
from perception_bev_learning.utils import (
    WeightedMeanSquaredError,
    WeightedMeanAbsoluteError,
    ValidMeanSquaredError,
    ValidMeanAbsoluteError,
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
        self.key = "bev_metric_"

        if self.meter_cfg.hdd_statistic:
            lm.pred_target_recall = torch.nn.ModuleDict()
            lm.pred_target_precision = torch.nn.ModuleDict()
            lm.pred_target_f1 = torch.nn.ModuleDict()

            lm.current_target_recall = torch.nn.ModuleDict()
            lm.current_target_precision = torch.nn.ModuleDict()
            lm.current_target_f1 = torch.nn.ModuleDict()

        if self.meter_cfg.mse:
            lm.pred_target__mse = torch.nn.ModuleDict()
            lm.pred_current__mse = torch.nn.ModuleDict()
            lm.current_target__mse = torch.nn.ModuleDict()

        if self.meter_cfg.mae:
            lm.pred_target__mae = torch.nn.ModuleDict()
            lm.pred_current__mae = torch.nn.ModuleDict()
            lm.current_target__mae = torch.nn.ModuleDict()

        if self.meter_cfg.cell_statistic:
            lm.pred_target__cell_stat = torch.nn.ModuleDict()
            lm.pred_current__cell_stat = torch.nn.ModuleDict()
            lm.current_target__cell_stat = torch.nn.ModuleDict()

        if self.meter_cfg.weighted_mse:
            lm.pred_target__wmse = torch.nn.ModuleDict()
            lm.pred_current__wmse = torch.nn.ModuleDict()
            lm.current_target__wmse = torch.nn.ModuleDict()

        if self.meter_cfg.weighted_mae:
            lm.pred_target__wmae = torch.nn.ModuleDict()
            lm.pred_current__wmae = torch.nn.ModuleDict()
            lm.current_target__wmae = torch.nn.ModuleDict()

        if self.meter_cfg.observed_vs_unobserved:
            lm.pred_target__observed_mae = torch.nn.ModuleDict()
            lm.current_target__observed_mae = torch.nn.ModuleDict()

            lm.pred_target__unobserved_mae = torch.nn.ModuleDict()
            lm.current_target__unobserved_mae = torch.nn.ModuleDict()

            lm.pred_target__observed_mse = torch.nn.ModuleDict()
            lm.current_target__observed_mse = torch.nn.ModuleDict()

            lm.pred_target__unobserved_mse = torch.nn.ModuleDict()
            lm.current_target__unobserved_mse = torch.nn.ModuleDict()

        self.reliable_aux_layer = meter_cfg.aux_layers.reliable.id
        self.current_idxs = []

        for layer in meter_cfg.target_layers.values():
            for m in meter_cfg.modes:
                if self.meter_cfg.hdd_statistic:
                    if layer.name == "wheel_risk":
                        lm.pred_target_precision[
                            self.key + layer.name + "_" + m
                        ] = Precision(average="none", task="binary", num_classes=2)
                        lm.pred_target_recall[self.key + layer.name + "_" + m] = Recall(
                            average="none", task="binary"
                        )
                        lm.pred_target_f1[self.key + layer.name + "_" + m] = F1Score(
                            average="none", task="binary"
                        )

                        lm.current_target_precision[
                            self.key + layer.name + "_" + m
                        ] = Precision(average="none", task="binary", num_classes=2)
                        lm.current_target_recall[
                            self.key + layer.name + "_" + m
                        ] = Recall(average="none", task="binary")
                        lm.current_target_f1[self.key + layer.name + "_" + m] = F1Score(
                            average="none", task="binary"
                        )

                if self.meter_cfg.mse:
                    lm.pred_target__mse[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanSquaredError()
                    lm.pred_current__mse[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanSquaredError()
                    lm.current_target__mse[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanSquaredError()

                if self.meter_cfg.mae:
                    lm.pred_target__mae[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanAbsoluteError()
                    lm.pred_current__mae[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanAbsoluteError()
                    lm.current_target__mae[
                        self.key + layer.name + "_" + m
                    ] = ValidMeanAbsoluteError()

                if self.meter_cfg.weighted_mse:
                    lm.pred_target__wmse[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanSquaredError()
                    lm.pred_current__wmse[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanSquaredError()
                    lm.current_target__wmse[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanSquaredError()

                if self.meter_cfg.weighted_mae:
                    lm.pred_target__wmae[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanAbsoluteError()
                    lm.pred_current__wmae[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanAbsoluteError()
                    lm.current_target__wmae[
                        self.key + layer.name + "_" + m
                    ] = WeightedMeanAbsoluteError()

                if layer.name == "wheel_risk":
                    f = f_se
                    compute_hdd = True and meter_cfg.compute_hdd_metric
                else:
                    f = f_ae
                    compute_hdd = False

                if self.meter_cfg.cell_statistic:
                    lm.pred_target__cell_stat[
                        self.key + layer.name + "_" + m
                    ] = CellStatistics(
                        loss_function=f, compute_hdd=compute_hdd, **asdict(meter_cfg)
                    )
                    lm.pred_current__cell_stat[
                        self.key + layer.name + "_" + m
                    ] = CellStatistics(
                        loss_function=f, compute_hdd=compute_hdd, **asdict(meter_cfg)
                    )
                    lm.current_target__cell_stat[
                        self.key + layer.name + "_" + m
                    ] = CellStatistics(
                        loss_function=f, compute_hdd=compute_hdd, **asdict(meter_cfg)
                    )

                if self.meter_cfg.observed_vs_unobserved:
                    if layer.name == "wheel_risk":
                        lm.pred_target__observed_mse[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanSquaredError()
                        lm.current_target__observed_mse[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanSquaredError()
                        lm.pred_target__unobserved_mse[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanSquaredError()
                        lm.current_target__unobserved_mse[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanSquaredError()
                    else:
                        lm.pred_target__observed_mae[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanAbsoluteError()
                        lm.current_target__observed_mae[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanAbsoluteError()
                        lm.pred_target__unobserved_mae[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanAbsoluteError()
                        lm.current_target__unobserved_mae[
                            self.key + layer.name + "_" + m
                        ] = ValidMeanAbsoluteError()

    def reset_cell_statistics(self, mode, lm):
        if self.meter_cfg.cell_statistic:
            for j, layer in enumerate(self.meter_cfg.target_layers.values()):
                lm.pred_target__cell_stat[self.key + layer.name + "_" + mode].reset()
                lm.pred_current__cell_stat[self.key + layer.name + "_" + mode].reset()
                lm.current_target__cell_stat[self.key + layer.name + "_" + mode].reset()

    @torch.no_grad()
    def update(self, target, pred, aux, mode, lm):
        m = mode
        if m not in self.meter_cfg.modes:
            return

        # perform cropping
        target = crop(target.detach())
        pred = crop(pred.detach())
        aux = crop(aux.detach())
        aux_reliable = aux[:, self.reliable_aux_layer]

        if self.meter_cfg.hdd_statistic:
            ln = "wheel_risk"
            if self.meter_cfg.target_layers.values()[0].name == ln:
                idx_target = 0
                idx_pred = 0
                idx_current = self.meter_cfg.target_layers.values()[0].aux_id

                mask = ~torch.isnan(target[:, idx_target]) * ~torch.isnan(
                    aux[:, idx_current]
                )
                print(self.meter_cfg.fatal_risk)
                b_target = target[:, idx_target] > self.meter_cfg.fatal_risk
                b_current = aux[:, idx_current] > self.meter_cfg.fatal_risk
                b_pred = pred[:, idx_pred] > self.meter_cfg.fatal_risk

                lm.pred_target_precision[self.key + ln + "_" + m](
                    b_pred[mask], b_target[mask]
                )
                lm.pred_target_recall[self.key + ln + "_" + m](
                    b_pred[mask], b_target[mask]
                )
                lm.pred_target_f1[self.key + ln + "_" + m](b_pred[mask], b_target[mask])

                lm.current_target_precision[self.key + ln + "_" + m](
                    b_current[mask], b_target[mask]
                )
                lm.current_target_recall[self.key + ln + "_" + m](
                    b_current[mask], b_target[mask]
                )
                lm.current_target_f1[self.key + ln + "_" + m](
                    b_current[mask], b_target[mask]
                )

            else:
                raise ValueError("Configuration is not as expected!")

        for j, layer in enumerate(self.meter_cfg.target_layers.values()):
            current = aux[:, layer.aux_id]

            # old_visu()
            if self.meter_cfg.observed_vs_unobserved:
                idx_reliable_current = [
                    j
                    for j, l in enumerate(self.meter_cfg.aux_layers)
                    if l.name == "reliable" and l.gridmap_topic == "gridmap_micro"
                ][0]
                idx_reliable_gt = [
                    j
                    for j, l in enumerate(self.meter_cfg.aux_layers)
                    if l.name == "reliable" and l.gridmap_topic == "gridmap_micro_gt"
                ][0]

                m_observed = aux[:, idx_reliable_current] > 0.5
                m_unobserved = (aux[:, idx_reliable_gt] > 0.5) * (~m_observed)

                if layer.name == "wheel_risk":
                    lm.pred_target__observed_mse[
                        self.key + layer.name + "_" + m
                    ].update(pred[:, j][m_observed], target[:, j][m_observed])
                    lm.current_target__observed_mse[
                        self.key + layer.name + "_" + m
                    ].update(current[m_observed], target[:, j][m_observed])
                    lm.pred_target__unobserved_mse[
                        self.key + layer.name + "_" + m
                    ].update(pred[:, j][m_unobserved], target[:, j][m_unobserved])
                    lm.current_target__unobserved_mse[
                        self.key + layer.name + "_" + m
                    ].update(current[m_unobserved], target[:, j][m_unobserved])
                else:
                    lm.pred_target__observed_mae[
                        self.key + layer.name + "_" + m
                    ].update(pred[:, j][m_observed], target[:, j][m_observed])
                    lm.current_target__observed_mae[
                        self.key + layer.name + "_" + m
                    ].update(current[m_observed], target[:, j][m_observed])
                    lm.pred_target__unobserved_mae[
                        self.key + layer.name + "_" + m
                    ].update(pred[:, j][m_unobserved], target[:, j][m_unobserved])
                    lm.current_target__unobserved_mae[
                        self.key + layer.name + "_" + m
                    ].update(current[m_unobserved], target[:, j][m_unobserved])

            if self.meter_cfg.mse:
                lm.pred_target__mse[self.key + layer.name + "_" + m](
                    pred[:, j], target[:, j]
                )
                lm.pred_current__mse[self.key + layer.name + "_" + m](
                    pred[:, j], current
                )
                lm.current_target__mse[self.key + layer.name + "_" + m](
                    current, target[:, j]
                )
                lm.log(
                    m + "_" + layer.name + "_pred_target__mse",
                    lm.pred_target__mse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_pred_current__mse",
                    lm.pred_current__mse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_current_target__mse",
                    lm.current_target__mse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
            if self.meter_cfg.mae:
                lm.pred_target__mae[self.key + layer.name + "_" + m](
                    pred[:, j], target[:, j]
                )
                lm.pred_current__mae[self.key + layer.name + "_" + m](
                    pred[:, j], current
                )
                lm.current_target__mae[self.key + layer.name + "_" + m](
                    current, target[:, j]
                )
                lm.log(
                    m + "_" + layer.name + "_pred_target__mae",
                    lm.pred_target__mae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_pred_current__mae",
                    lm.pred_current__mae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_current_target__mae",
                    lm.current_target__mae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )

            if self.meter_cfg.weighted_mse:
                lm.pred_target__wmse[self.key + layer.name + "_" + m](
                    pred[:, j], target[:, j], aux_reliable
                )
                lm.pred_current__wmse[self.key + layer.name + "_" + m](
                    pred[:, j], current, aux_reliable
                )
                lm.current_target__wmse[self.key + layer.name + "_" + m](
                    current, target[:, j], aux_reliable
                )
                lm.log(
                    m + "_" + layer.name + "_pred_target__wmse",
                    lm.pred_target__wmse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_pred_current__wmse",
                    lm.pred_current__wmse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_current_target__wmse",
                    lm.current_target__wmse[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )

            if self.meter_cfg.weighted_mae:
                lm.pred_target__wmae[self.key + layer.name + "_" + m](
                    pred[:, j], target[:, j], aux_reliable
                )
                lm.pred_current__wmae[self.key + layer.name + "_" + m](
                    pred[:, j], current, aux_reliable
                )
                lm.current_target__wmae[self.key + layer.name + "_" + m](
                    current, target[:, j], aux_reliable
                )
                lm.log(
                    m + "_" + layer.name + "_pred_target__wmae",
                    lm.pred_target__wmae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_pred_current__wmae",
                    lm.pred_current__wmae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )
                lm.log(
                    m + "_" + layer.name + "_current_target__wmae",
                    lm.current_target__wmae[self.key + layer.name + "_" + m],
                    on_epoch=True,
                    on_step=False,
                )

            if self.meter_cfg.cell_statistic:
                lm.pred_target__cell_stat[self.key + layer.name + "_" + m].update(
                    pred[:, j], target[:, j]
                )
                lm.pred_current__cell_stat[self.key + layer.name + "_" + m].update(
                    pred[:, j], current
                )
                lm.current_target__cell_stat[self.key + layer.name + "_" + m].update(
                    current, target[:, j]
                )


# def old_visu():
# TODO integrate that the visualization works and is meaningfull if needed
# from PIL import Image
# i = current[j].cpu().numpy()
# i -= i[~np.isnan(i)].min()
# i /= i[~np.isnan(i)].max()
# i *= 256
# i = np.uint8(i)

# img = Image.fromarray(i)
# img.show()

# Without NN the comparision to the existing stack is
# Alternative:
# - Report comparision to existing stack on the valid data only
# - plus the comparision to the NN extrapolation

# clone_cur = current.clone()

#### VISU INPUT ###
# i = current[j].cpu().numpy()
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# from perception_bev_learning.visu import get_img_from_fig
# cmap = sns.color_palette("viridis", as_cmap=True)
# cmap.set_bad(color="black")
# fig = plt.figure(figsize=(5,5))
# plt.imshow(i ,cmap= cmap, vmin=i[~np.isnan(i)].min(), vmax=i[~np.isnan(i)].max())
# plt.show()
# res = get_img_from_fig(fig)
# res.show()

#### Sketchy Nearest Neighbour interpolation INPUT ###
# from scipy.interpolate import NearestNDInterpolator
# not_filled_current = (torch.isnan( current ) * ~torch.isnan( target[:, j] ))
# filled_current = ~torch.isnan( current )
# indices = torch.where(filled_current)
# indices_not_filled = np.where(not_filled_current.cpu().numpy())
# indices_filled = np.where(filled_current.cpu().numpy())
# nndi = NearestNDInterpolator(indices_filled, current[filled_current].cpu().numpy())
# res = nndi(indices_not_filled)
# current[ torch.where(not_filled_current) ] = torch.from_numpy(res).to(current.device)

### Better interpolation
# from pytictac import Timer
# with Timer("test"):
#     from scipy.interpolate import CloughTocher2DInterpolator
#     for bs in range(current.shape[0]):
#         not_filled_current = (torch.isnan( current[bs] ) * ~torch.isnan( target[bs, j] ))
#         filled_current = ~torch.isnan( current[bs] )
#         indices = torch.where(filled_current)
#         indices_not_filled = np.where(not_filled_current.cpu().numpy())
#         indices_filled = np.where(filled_current.cpu().numpy())
#         interp = CloughTocher2DInterpolator(indices_filled, current[bs][filled_current].cpu().numpy())
#         res = interp(indices_not_filled[0], indices_not_filled[1])
#         current[bs][torch.where(not_filled_current) ] = torch.from_numpy(res).to(current.device).type(current.dtype)

# Plot modified current
# i = current[j].cpu().numpy()
# cmap = sns.color_palette("viridis", as_cmap=True)
# cmap.set_bad(color="black")
# fig = plt.figure(figsize=(5,5))
# plt.imshow(i ,cmap= cmap, vmin=i[~np.isnan(i)].min(), vmax=i[~np.isnan(i)].max())
# plt.show()
# res1 = get_img_from_fig(fig)
# res1.show()
