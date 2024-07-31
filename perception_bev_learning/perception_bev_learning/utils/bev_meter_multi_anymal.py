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
from torchmetrics.segmentation import MeanIoU

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


class BevMeterMultiAnymal(nn.Module):
    def __init__(self, meter_cfg, lm, cfg=None):
        super().__init__()
        self.meter_cfg = meter_cfg
        self.key = "bm_"

        self.target_idxs_dict = {}
        self.aux_idxs_dict = {}

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

            lm.pred_target__miou = torch.nn.ModuleDict()
            lm.pred_target__mae = torch.nn.ModuleDict()

        for gridmap_key in meter_cfg.target_layers.keys():
            for lname, layer in meter_cfg.target_layers[gridmap_key].layers.items():
                for m in meter_cfg.modes:
                    if lname == "traversability":
                        lm.pred_target__miou[
                            gridmap_key + lname + "_" + m
                        ] = MeanIoU(num_classes=3, per_class=False, include_background=False)

                    if lname == "elevation":
                        lm.pred_target__mae[
                            gridmap_key + lname + "_" + m
                        ] = ValidMeanAbsoluteError()


    @torch.no_grad()
    def update(self, target, pred, aux, mode, lm):
        m = mode
        if m not in self.meter_cfg.modes:
            return

        for gridmap_key in self.meter_cfg.target_layers.keys():
            # perform cropping
            target_c = target[gridmap_key].detach()
            pred_c = pred[gridmap_key].detach()
            aux_c = aux[gridmap_key].detach()

            for j, (lname, layer) in enumerate(
                self.meter_cfg.target_layers[gridmap_key].layers.items()
            ):

                if lname == "traversability":
                    # Need to convert target to binary values
                    pred_trav = pred_c[:, j]
                    target_trav = target_c[:, j]
                    m_valid = ~torch.isnan(target_trav)

                    target_segmented = torch.zeros_like(target_trav, dtype=torch.long)
                    target_segmented[(target_trav < 0.5) & m_valid] = 1
                    target_segmented[(target_trav >= 0.5) & m_valid] = 2

                    pred_segmented = torch.zeros_like(pred_trav, dtype=torch.long)
                    pred_segmented[(pred_trav < 0.5) & m_valid] = 1
                    pred_segmented[(pred_trav >= 0.5) & m_valid] = 2

                    lm.pred_target__miou[gridmap_key + lname + "_" + m](
                        pred_segmented,
                        target_segmented
                    )
                    lm.log(f"{m}/{gridmap_key}/{lname}/Pred/mIOU",
                        lm.pred_target__miou[gridmap_key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,)

                if lname == "elevation":
                    lm.pred_target__mae[gridmap_key + lname + "_" + m](
                        pred_c[:, j],
                        target_c[:, j]
                    )
                    lm.log(
                        f"{m}/{gridmap_key}/{lname}/Pred/Total_MAE",
                        lm.pred_target__mae[gridmap_key + lname + "_" + m],
                        on_epoch=True,
                        on_step=False,
                    )