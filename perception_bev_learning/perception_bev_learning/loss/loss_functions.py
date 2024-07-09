import torch
from torch.nn import functional as F
from os.path import join

EPS = 0.001


def valid_mse(pred, target):
    m = ~(torch.isnan(target))
    return F.mse_loss(pred[m], target[m], reduction="none").mean()


def mse_wheel_risk_balanced(pred, target, weight):
    m = ~(torch.isnan(target))
    res = F.mse_loss(pred[:, 0][m], target[m], reduction="none")
    idx = (target[m] * 99).type(torch.long)
    if weight is not None:
        weight = weight.to(target.device)
        res = res * weight[idx]

    return res.mean()


def elevation_fused_l1_residual(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    weight_unobs = 0.2

    raw_ele = aux[:, aux_dict["elevation_raw"]][:, None]
    raw_ele[torch.isnan(raw_ele)] = 0
    pred = pred + raw_ele

    m_obs = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = ~(torch.isnan(target)) * ~(torch.isnan(pred))

    loss_obs = F.smooth_l1_loss(pred[m_obs * m], target[m_obs * m], reduction="none")
    loss_unobs = F.smooth_l1_loss(
        pred[~m_obs * m], target[~m_obs * m], reduction="none"
    )

    res = loss_obs.mean() + weight_unobs * loss_unobs.mean()

    return res


def elevation_fused_l1(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    weight_unobs = 0.2

    m_obs = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = ~(torch.isnan(target)) * ~(torch.isnan(pred))

    loss_obs = F.smooth_l1_loss(pred[m_obs * m], target[m_obs * m], reduction="none")
    loss_unobs = F.smooth_l1_loss(
        pred[~m_obs * m], target[~m_obs * m], reduction="none"
    )

    # # Gradient Matching loss
    # mask = m * m_obs
    # weight_grad = 0.5
    # N = torch.sum(mask)
    # diff = pred - target
    # diff = torch.mul(diff, mask)

    # # print(f"Mask shape is {mask.shape}")
    # # print(f"diff shape is {diff.shape}")

    # v_gradient = torch.abs(diff[..., 0:-2, :] - diff[..., 2:, :])
    # v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
    # v_gradient = torch.mul(v_gradient, v_mask)

    # # print(f"v_grad shape is {v_gradient.shape}")

    # h_gradient = torch.abs(diff[..., :, 0:-2] - diff[..., :, 2:])
    # h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
    # h_gradient = torch.mul(h_gradient, h_mask)

    # nan_mask_h = torch.isnan(h_gradient)
    # nan_mask_v = torch.isnan(v_gradient)

    # h_gradient[nan_mask_h] = 0
    # v_gradient[nan_mask_v] = 0

    # gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    # gradient_loss = gradient_loss / N

    res = loss_obs.mean() + weight_unobs * loss_unobs.mean()

    return res


def elevation_fused_gnll(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,2,H,W)): Prediction (Mean and SD)
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    weight_unobs = 0.5

    m_obs = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = ~(torch.isnan(target)) * ~(torch.isnan(pred[:, 0][:, None]))

    loss_obs = F.gaussian_nll_loss(
        pred[:, 0][:, None][m_obs * m],
        target[m_obs * m],
        pred[:, 1][:, None][m_obs * m],
        reduction="none",
    )
    loss_unobs = F.gaussian_nll_loss(
        pred[:, 0][:, None][~m_obs * m],
        target[~m_obs * m],
        pred[:, 1][:, None][~m_obs * m],
        reduction="none",
    )

    res = loss_obs.mean() + weight_unobs * loss_unobs.mean()

    return res


def elevation_fused_uncertainty(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,2,H,W)): Prediction (Mean and SD)
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    weight_unobs = 0.5
    eps = 1e-6
    m_obs = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = ~(torch.isnan(target)) * ~(torch.isnan(pred[:, 0][:, None]))

    loss_obs = pred[:, 1][:, None][m_obs * m] + F.smooth_l1_loss(
        pred[:, 0][:, None][m_obs * m], target[m_obs * m], reduction="none"
    ) / torch.max(pred[:, 1][:, None][m_obs * m], torch.tensor(eps, device=pred.device))

    loss_unobs = pred[:, 1][:, None][~m_obs * m] + F.smooth_l1_loss(
        pred[:, 0][:, None][~m_obs * m], target[~m_obs * m], reduction="none"
    ) / torch.max(
        pred[:, 1][:, None][~m_obs * m], torch.tensor(eps, device=pred.device)
    )

    res = loss_obs.mean() + weight_unobs * loss_unobs.mean()

    return res


def pre_gnll(pred, *args, **kwargs):
    pred[:, 1][:, None] = F.relu(pred[:, 1][:, None])

    return pred


def residual_elevation(pred, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers

    Compute the elevation prediction by adding residuals to the raw elevation
    """
    raw_ele = aux[:, aux_dict["elevation_raw"]][:, None]
    raw_ele[torch.isnan(raw_ele)] = 0
    elevation = pred + raw_ele

    return elevation


def wheel_risk_mse(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    m_valid = ~(torch.isnan(target)) * ~(torch.isnan(pred))
    m_conf = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = m_conf * m_valid

    res = F.mse_loss(pred[m], target[m], reduction="none")

    return res.mean()


def wheel_risk_bce(pred, target, aux, aux_dict):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    aux (torch.Tensor shape:=(BS,N,H,W)): Aux layers
    aux_dict (dict): Dictionary containing the keys as layer names and values as indices for aux
    """
    lethal_threshold = 0.5
    m_valid = ~(torch.isnan(target)) * ~(torch.isnan(pred))
    m_conf = aux[:, aux_dict["confidence_gt"]][:, None] > 0.5
    m = m_conf * m_valid

    target = torch.where(target > lethal_threshold, 1, 0).float()

    res = F.binary_cross_entropy(pred[m], target[m], reduction="none")

    return res.mean()


def mse_regression(pred, target, bin_low=None, bin_high=None, weight=None, *args):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    """
    m = ~(torch.isnan(target))
    res = F.mse_loss(pred[m], target[m], reduction="none")
    if weight is not None:
        with torch.no_grad():
            weight_idx = target[m].clip(bin_low, bin_high) - bin_low
            weight_idx /= bin_high - bin_low + EPS
            weight_idx *= weight.shape[0]
            weight_idx = weight_idx.type(torch.long)
            weight = weight.to(target.device)
        res = res * weight[weight_idx]

    return res.mean()


def bin_classification(pred, target, bin_low=-1, bin_high=1, weight=None):
    """
    pred (torch.Tensor shape:=(BS,1,H,W)): Prediction
    target (torch.Tensor shape:=(BS,1,H,W)): Target
    """

    m = torch.isnan(target)
    with torch.no_grad():
        target_bins = target.clip(bin_low, bin_high) - bin_low
        target_bins /= bin_high - bin_low + EPS

        nr_bins = pred.shape[1]
        target_bins *= nr_bins

        target_bins = target_bins.type(torch.long)
        target_bins[m] = -1

    if weight is not None:
        weight = weight.to(target.device)
    res = F.cross_entropy(
        pred,
        target_bins[:, 0],
        weight=weight,
        ignore_index=-1,
        reduction="none",
        label_smoothing=0.0,
    )
    return res.mean()


def bin_to_value(pred, bin_low=-1, bin_high=1):
    return (
        bin_low + torch.argmax(pred, dim=1) / pred.shape[1] * (bin_high - bin_low)
    ).unsqueeze(1)


def skip(pred, *args, **kwargs):
    return pred


def compute_histogramm(pred):
    import matplotlib

    matplotlib.use("agg")
    from torch import nn
    import matplotlib.pyplot as plt

    val = torch.softmax(pred, dim=1).sum(axis=[2, 3]).reshape(-1)
    fig, ax = plt.subplots()

    labels = [f"bin_{x}" for x in range(val.shape[0])]
    counts = val.cpu().numpy()
    ax.bar(labels, counts, label=labels)

    ax.set_ylabel("fruit supply")
    ax.set_title("Fruit supply by kind and color")
    ax.legend(title="Fruit color")

    from perception_bev_learning.visu import get_img_from_fig

    img = get_img_from_fig(fig)
    import imageio

    imageio.imwrite("/tmp/img.png", img)

    plt.show()
