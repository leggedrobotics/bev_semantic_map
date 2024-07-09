from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.mse import (
    _mean_squared_error_compute,
    _mean_squared_error_update,
)
from torchmetrics.metric import Metric
from torch.nn.functional import mse_loss
from torchmetrics import Precision, Recall, F1Score
from pytictac import Timer


def f_ae(preds, target):
    return torch.abs(preds - target).type(torch.float64)


def f_se(preds, target):
    return ((preds - target) ** 2).type(torch.float64)


class CellStatistics(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    sum_squared_error: Tensor
    sum_error: Tensor
    total: Tensor

    def __init__(
        self,
        bev_map_size,
        loss_function,
        compute_hdd,
        fatal_risk,
        grid_resolution,
        bin_distance_in_m,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.bev_map_size = bev_map_size
        self.fatal_risk = fatal_risk
        self.grid_resolution = grid_resolution
        self.bin_distance_in_m = bin_distance_in_m
        self.compute_hdd = compute_hdd
        self.init_buffers()

        # Calculate the center pixel coordinates
        center_x = (bev_map_size[0] - 1) / 2  # Since indexing starts from 0
        center_y = (bev_map_size[1] - 1) / 2  # Since indexing starts from 0
        # Create a grid of coordinates representing each pixel in the image
        x = torch.arange(0, bev_map_size[0]) * grid_resolution - (
            center_x * grid_resolution
        )
        y = torch.arange(0, bev_map_size[1]) * grid_resolution - (
            center_y * grid_resolution
        )
        # Create a meshgrid from x and y coordinates
        X, Y = torch.meshgrid(x, y)
        # Calculate the distance from each pixel to the center pixel
        distance_to_center = torch.sqrt(X**2 + Y**2)
        mapping = torch.full(bev_map_size, -1, dtype=torch.long)

        for i in range(self.nr_bins):
            mapping[
                (distance_to_center > i * bin_distance_in_m)
                * (distance_to_center < (i + 1) * bin_distance_in_m)
            ] = i

        self.add_state("mapping", default=mapping, dist_reduce_fx=None)

    def init_buffers(self):
        # Add Buffer for the mean maps
        self.add_state(
            "sum_squared_error",
            default=torch.zeros(self.bev_map_size, dtype=torch.float64),
            dist_reduce_fx=None,
        )
        self.add_state(
            "sum_error",
            default=torch.zeros(self.bev_map_size, dtype=torch.float64),
            dist_reduce_fx=None,
        )
        self.add_state(
            "total",
            default=torch.zeros(self.bev_map_size, dtype=torch.float64),
            dist_reduce_fx=None,
        )

        self.nr_bins = (
            int(
                self.bev_map_size[0]
                / 2
                * self.grid_resolution
                * 2**0.5
                / self.bin_distance_in_m
            )
            + 1
        )
        self.add_state(
            "sum_squared_error_bins",
            default=torch.zeros(self.nr_bins, dtype=torch.float64),
            dist_reduce_fx=None,
        )
        self.add_state(
            "sum_error_bins",
            default=torch.zeros(self.nr_bins, dtype=torch.float64),
            dist_reduce_fx=None,
        )
        self.add_state(
            "total_bins",
            default=torch.zeros(self.nr_bins, dtype=torch.float64),
            dist_reduce_fx=None,
        )

        if self.compute_hdd:
            # Add the buffer for the hazards statistics
            self.precision_bins = torch.nn.ModuleList(
                [
                    Precision(average="none", task="binary", num_classes=2)
                    for i in range(self.nr_bins)
                ]
            )
            self.f1_bins = torch.nn.ModuleList(
                [
                    F1Score(average="none", task="binary", num_classes=2)
                    for i in range(self.nr_bins)
                ]
            )
            self.recall_bins = torch.nn.ModuleList(
                [
                    Recall(average="none", task="binary", num_classes=2)
                    for i in range(self.nr_bins)
                ]
            )

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        for b in range(preds.shape[0]):
            m_target = ~torch.isnan(target[b])
            # m_preds = ~torch.isnan(preds[b])
            m = m_target  # * m_preds
            if m.sum() > 0:
                error = self.loss_function(preds[b][m], target[b][m])

                # Simply add to each gridcell the error - can be done simply
                self.sum_error[m] += error
                self.sum_squared_error[m] += torch.square(error)
                self.total[m] += 1

                # Apply the typicial cumsum trick to estimate the sum
                ma_ori = self.mapping[m]
                sorts = ma_ori.argsort()
                ma = ma_ori[sorts]
                error = error[sorts]

                kept = torch.ones(error.shape[0], device=error.device, dtype=torch.bool)
                kept[:-1] = ma[1:] != ma[:-1]
                error = torch.cumsum(error, dim=0)
                error = error[kept]
                error = torch.cat((error[:1], error[1:] - error[:-1]))
                res = torch.where(kept)[0]
                bin_count = torch.cat((res[:1], res[1:] - res[:-1]))
                bin_indices = ma[kept]

                self.sum_error_bins[bin_indices] += error
                self.sum_squared_error_bins[bin_indices] += torch.square(error)
                self.total_bins[bin_indices] += bin_count

                # Compute the classification metrics for fatal_risk
                if self.compute_hdd:
                    b_pred = preds[b][m] > self.fatal_risk
                    b_target = target[b][m] > self.fatal_risk
                    # This is very slow - is there a faster implementation
                    # The only way to make this faster also use cumsum trick here and write metric on your own
                    for i in range(self.nr_bins):
                        m2 = ma_ori == i
                        if m2.sum() > 0:
                            self.precision_bins[i](
                                preds=b_pred[m2], target=b_target[m2]
                            )
                            self.f1_bins[i](preds=b_pred[m2], target=b_target[m2])
                            self.recall_bins[i](preds=b_pred[m2], target=b_target[m2])

    def get_maps(self):
        return_mean = torch.zeros_like(self.sum_error)
        return_variance = torch.zeros_like(self.sum_error)

        m = self.total != 0
        if m.sum() > 0:
            return_mean[m] = self.sum_error[m] / self.total[m]
            return_variance[m] = self.sum_squared_error[m] / self.total[
                m
            ] - torch.square(return_mean[m])

        return_mean[~m] = torch.nan
        return_variance[~m] = torch.nan

        return return_mean, return_variance

    def get_bins(self):
        return_mean = torch.zeros_like(self.sum_error_bins)
        return_variance = torch.zeros_like(self.sum_error_bins)

        m = self.total_bins != 0
        if m.sum() > 0:
            return_mean[m] = self.sum_error_bins[m] / self.total_bins[m]
            return_variance[m] = self.sum_squared_error_bins[m] / self.total_bins[
                m
            ] - torch.square(return_mean[m])

        return_mean[~m] = torch.nan
        return_variance[~m] = torch.nan
        return return_mean, return_variance

    def reset(self):
        self.init_buffers()

        self.sum_squared_error.zero_()
        self.sum_error.zero_()
        self.total.zero_()

        self.sum_squared_error_bins.zero_()
        self.sum_error_bins.zero_()
        self.total_bins.zero_()

        if self.compute_hdd:
            [
                metric.reset()
                for metric in self.precision_bins + self.f1_bins + self.recall_bins
            ]

    def compute(self) -> Tensor:
        return (self.sum_squared_error.sum() / self.total.sum()).type(torch.float32)


if __name__ == "__main__":
    # Test metric
    bev_map_size = (1, 500, 500)
    metric = CellStatistics(bev_map_size=(500, 500))
    for i in range(100):
        preds = torch.rand(bev_map_size)
        targets = torch.rand(bev_map_size)

        preds[:10, :10] = 0
        targets[-10:, -20:] = 0
        metric.update(preds, targets)

    return_mean, return_variance = metric.get_maps()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(return_mean.cpu().detach().numpy(), cmap="plasma")
    plt.show()


class ValidMeanSquaredError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_squared_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        m = ~torch.isnan(target) * ~torch.isnan(preds)
        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_squared_error = torch.sum(diff * diff)
            self.sum_squared_error += sum_squared_error
            self.total += m.sum()

    def compute(self) -> Tensor:
        return (self.sum_squared_error / self.total).type(torch.float32)


class MaskedMeanSquaredError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_squared_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        m = mask * ~torch.isnan(preds) * ~torch.isnan(target)
        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_squared_error = torch.sum(diff * diff)
            self.sum_squared_error += sum_squared_error
            self.total += m.sum()

    def compute(self) -> Tensor:
        return (self.sum_squared_error / self.total).type(torch.float32)


class ValidMeanMetric(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        m = ~torch.isnan(target) * ~torch.isnan(preds) * mask
        m_total = torch.ones_like(mask)

        self.sum += m.sum()
        self.total += m_total.sum()

    def compute(self) -> Tensor:
        return (self.sum / self.total).type(torch.float32)


class ValidMeanAbsoluteError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_mae_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_mae_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        m = ~torch.isnan(target) * ~torch.isnan(preds)
        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_mae_error = torch.sum(torch.abs(diff))
            self.sum_mae_error += sum_mae_error
            self.total += m.sum()

    def compute(self) -> Tensor:
        return (self.sum_mae_error / self.total).type(torch.float32)


class MaskedMeanAbsoluteError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_mae_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_mae_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:  # type: ignore
        m = mask * ~torch.isnan(preds) * ~torch.isnan(target)
        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_mae_error = torch.sum(torch.abs(diff))
            self.sum_mae_error += sum_mae_error
            self.total += m.sum()

    def compute(self) -> Tensor:
        return (self.sum_mae_error / self.total).type(torch.float32)


class WeightedMeanSquaredError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_squared_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, weights: Tensor) -> None:  # type: ignore
        m_target = ~torch.isnan(target)
        m_weights = ~torch.isnan(weights)
        m_preds = ~torch.isnan(preds)
        m = m_target * m_weights * m_preds

        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_squared_error = torch.sum((diff * diff) * weights[m])
            self.sum_squared_error += sum_squared_error
            self.total += (weights[m]).sum().type(torch.long)

    def compute(self) -> Tensor:
        return (self.sum_squared_error / self.total).type(torch.float32)


class WeightedMeanAbsoluteError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_abs_error",
            default=tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, weights: Tensor) -> None:  # type: ignore
        m_target = ~torch.isnan(target)
        m_weights = ~torch.isnan(weights)
        m_preds = ~torch.isnan(preds)
        m = m_target * m_weights * m_preds

        if m.sum() != 0:
            diff = preds[m] - target[m]
            sum_abs_error = torch.abs((diff) * weights[m]).sum()
            self.sum_abs_error += sum_abs_error
            self.total += (weights[m]).sum().type(torch.long)

    def compute(self) -> Tensor:
        return (self.sum_abs_error / self.total).type(torch.float32)
