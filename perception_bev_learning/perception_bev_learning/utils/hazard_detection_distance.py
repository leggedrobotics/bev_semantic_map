import torch
import numpy as np
from scipy.ndimage import label
import copy

EPS = 1e-6


def connected_components(binary_mask: torch.Tensor):
    """Function to perform connected component labeling
    Args:
        binary_mask (torch.tensor, dtype=torch.bool, shape(H,W)): binary mask of hazards
    Returns:
        label_mask (torch.tensor, dtype=torch.long, shape (H,W)): segmentation mask of hazards
    """

    labeled_mask, num_labels = label(binary_mask.cpu())
    return torch.from_numpy(labeled_mask).to(binary_mask.device)


def process_tensor(
    risk_map: torch.Tensor, fatal_risk: float, dilation_kernel_size: int
):
    binary_mask = risk_map > fatal_risk
    # Dilation operation
    kernel = torch.ones(
        1, 1, dilation_kernel_size, dilation_kernel_size, device=risk_map.device
    )
    binary_mask = (
        torch.nn.functional.conv2d(
            binary_mask[None, None].type(torch.float32),
            kernel,
            padding=dilation_kernel_size // 2,
        )[0, 0]
        > 0
    )
    # Connected componnent labeling
    components = connected_components(binary_mask)
    return components


def get_distance(center_u, center_v, binary: torch.Tensor, resolution: float):
    H, W = binary.shape
    Hc = int(H / 2)
    Wc = int(W / 2)
    return ((center_u - Hc) ** 2 + (center_v - Wc) ** 2) ** 0.5


def compute_iou(binary_pred: torch.Tensor, binary_target: torch.Tensor):
    """Compute IoU between binary masks

    Args:
        binary_pred (shape:=(H,W)): _description_
        binary_target (shape:=(N,H,W)): _description_

    Returns:
        torch.tensor, shape:=(N): IoU valiues
    """

    # Compute intersection and union for each pair of masks in the batch
    intersection = torch.sum(binary_pred & binary_target, dim=(1, 2))
    union = torch.sum(binary_pred | binary_target, dim=(1, 2))
    iou = (intersection.float()) / (union.float() + EPS)
    return iou


def iou_binary_to_segments(segments: torch.Tensor, binary: torch.Tensor):
    segments_idxs = torch.unique(segments[~torch.isnan(segments)])
    # Remove the 0 index from the segments
    segments_idxs = segments_idxs[segments_idxs != 0]
    nr_segments = segments_idxs.shape[0]

    h, w = binary.shape
    # bibinary[None].repeat( nr_segments, 1, 1)
    segments_expanded = torch.zeros(
        (nr_segments, h, w), dtype=bool, device=binary.device
    )
    for i, j in enumerate(segments_idxs):
        segments_expanded[i][segments == j] = True

    iou_values = compute_iou(binary, segments_expanded)

    return iou_values, segments_idxs


def greedy_matching(pred: torch.Tensor, target: torch.Tensor, threshold: float):
    """Iterate over each target index and find the best correspnding prediction segment based on IoU

    Args:
        pred (torch.tensor, shape:=(H,W)): _description_
        target (torch.tensor. shape:=(H,W)): _description_
        threshold (float): IoU threshold value for successfull assignement - strictly larger 0

    Returns:
        _type_: _description_
    """

    target_idxs = torch.unique(target[~torch.isnan(target)])
    results = []

    for i in target_idxs:
        if i == 0:
            continue
        binary = target == i
        ious, segment_idxs = iou_binary_to_segments(segments=pred, binary=binary)

        # Filter the matches based on IOU score
        segment_idxs = segment_idxs[ious > threshold]
        ious = ious[ious > threshold]

        # Compute the target hazard distance
        u, v = torch.where(binary)
        center_u, center_v = u.type(torch.float32).mean(), v.type(torch.float32).mean()
        dis = get_distance(center_u, center_v, binary, resolution=0.2).item()

        res = {"target_idx": i.item(), "dis_aux": dis, "dis_target": dis}

        if len(ious) > 0:
            for j, iou in enumerate(ious):
                res["pred_idx"] = segment_idxs[j].item()
                res["iou"] = iou.item()
                # Compute the predicted hazard distance
                binary_pred = pred == segment_idxs[j]
                u, v = torch.where(binary_pred)
                center_u, center_v = (
                    u.type(torch.float32).mean(),
                    v.type(torch.float32).mean(),
                )
                dis = get_distance(
                    center_u, center_v, binary_pred, resolution=0.2
                ).item()
                res["dis_pred"] = dis
                res["type"] = "true_positive"
                results.append(copy.deepcopy(res))
        else:
            res["type"] = "false_negative"
            res["pred_idx"] = -1
            res["dis_pred"] = -1
            results.append(copy.deepcopy(res))

    # Label all predicted hazards that are not matched to a target hazard as a false_postive sample
    used_preds = torch.unique(
        torch.tensor([r["pred_idx"] for r in results], device=pred.device)
    )
    for pred_idx in torch.unique(pred):
        if pred_idx not in used_preds:
            if pred_idx == 0:
                continue
            binary_pred = pred == pred_idx
            binary_pred[binary_pred].sum() != 0

            u, v = torch.where(binary_pred)
            center_u, center_v = (
                u.type(torch.float32).mean(),
                v.type(torch.float32).mean(),
            )
            dis = get_distance(center_u, center_v, binary_pred, resolution=0.2).item()
            results.append(
                {
                    "target_idx": -1,
                    "pred_idx": pred_idx.item(),
                    "iou": 0,
                    "type": "false_positive",
                    "dis_aux": dis,
                    "dis_pred": dis,
                    "dis_target": -1,
                }
            )

    return results


def generate_matching_statistics(hdd_matches, max_distance=150, distance_binning=10):
    s = len(hdd_matches)
    target_idx = torch.zeros((s,), dtype=torch.long)
    dis_aux = torch.zeros((s,), dtype=torch.float32)
    dis_target = torch.zeros((s,), dtype=torch.float32)
    pred_idx = torch.zeros((s,), dtype=torch.long)
    dis_pred = torch.zeros((s,), dtype=torch.float32)
    matching_type = torch.zeros((s,), dtype=torch.long)
    mapping = {
        "false_negative": 0,
        "false_positive": 1,
        "true_positive": 2,
        "true_negative": 3,
    }

    for s in range(s):
        dis_aux[s] = hdd_matches[s]["dis_aux"]
        dis_target[s] = hdd_matches[s]["dis_target"]
        pred_idx[s] = hdd_matches[s]["pred_idx"]
        dis_pred[s] = hdd_matches[s]["dis_pred"]
        matching_type[s] = mapping[hdd_matches[s]["type"]]

    TP = (matching_type == 2).sum()
    FN = (matching_type == 0).sum()
    hazard_detection_ratio = (TP / (TP + FN)).item()

    # dis_target = [m["dis_target"] for m in matching_result]
    # dis_pred = [m["dis_pred"] for m in matching_result if "dis_pred" in m]

    lower = 0
    bins = {
        "hazard_detection_ratio": [],
        "false_hazard_ratio": [],
        "lower": [],
        "upper": [],
    }

    for upper in np.arange(distance_binning, max_distance, distance_binning):
        TP = ((matching_type == 2) * (dis_aux >= lower) * (dis_aux < upper)).sum()
        FN = ((matching_type == 0) * (dis_aux >= lower) * (dis_aux < upper)).sum()
        FP = ((matching_type == 1) * (dis_aux >= lower) * (dis_aux < upper)).sum()

        if TP + FN != 0:
            bins["hazard_detection_ratio"].append((TP / (TP + FN)).item())
        else:
            bins["hazard_detection_ratio"].append(-1)
        if TP + FP != 0:
            bins["false_hazard_ratio"].append((FP / (TP + FP)).item())
        else:
            bins["false_hazard_ratio"].append(-1)

        bins["lower"].append(lower)
        bins["upper"].append(upper)

        lower = upper

    cumulative = {
        "hazard_detection_ratio": [],
        "false_hazard_ratio": [],
        "dis_aux": [],
    }

    so = np.argsort(dis_aux)

    dis_aux = dis_aux[so]
    for j, upper in enumerate(dis_aux):
        TP = ((matching_type == 2) * (dis_aux < upper)).sum()
        FN = ((matching_type == 0) * (dis_aux < upper)).sum()
        FP = ((matching_type == 1) * (dis_aux < upper)).sum()
        # If this is not quick not needed to check for all and we can just index

        cumulative["dis_aux"].append(upper.item())
        if TP + FN != 0:
            cumulative["hazard_detection_ratio"].append((TP / (TP + FN)).item())
        else:
            cumulative["hazard_detection_ratio"].append(-1)
        if TP + FP != 0:
            cumulative["false_hazard_ratio"].append((FP / (TP + FP)).item())
        else:
            cumulative["false_hazard_ratio"].append(-1)

    return hazard_detection_ratio, cumulative, bins


if __name__ == "__main__":
    """
    Script used to compute the statistics across the training dataset.
    Mainly used to create a balanced loss function!
    """
    from perception_bev_learning.dataset import get_bev_dataloader
    from perception_bev_learning.cfg import ExperimentParams
    from perception_bev_learning.utils import denormalize_img
    from perception_bev_learning.visu import LearningVisualizer, show
    from perception_bev_learning.visu import show, get_img_from_fig
    from pytorch_lightning import seed_everything
    import matplotlib.pyplot as plt

    seed_everything(42)
    cfg = ExperimentParams()
    cfg.update()
    cfg.dataloader_train.batch_size = 1
    loader_train, loader_val, loader_test = get_bev_dataloader(cfg)

    visu = LearningVisualizer()

    matching_result = []
    for j, batch in enumerate(loader_train):
        (imgs, rots, trans, intrins, post_rots, post_trans, target, *_) = batch
        # components = process_tensor(target[0,0], fatal_risk=0.2, dilation_kernel_size=3)
        aux = batch[-4]
        # show(img)
        risk_target = process_tensor(
            target[0, 0], fatal_risk=0.5, dilation_kernel_size=8
        )
        risk_current = process_tensor(aux[0, 0], fatal_risk=0.5, dilation_kernel_size=8)

        img1 = visu.plot_segmentation(risk_target, max_seg=risk_target.max() + 1)
        img2 = visu.plot_segmentation(risk_current, max_seg=risk_current.max() + 1)

        show(img1)
        show(img2)

        ls = greedy_matching(pred=risk_current, target=risk_target, threshold=0.1)

        matching_result += ls
        hazard_detection_ratio, cumulative, bins = generate_matching_statistics(
            matching_result, distance_binning=5
        )

        # Extract x and y values for plotting
        x_values = cumulative["dis_aux"]  # [item for item in cumulative]
        y_hazard_detection_ratio = cumulative["hazard_detection_ratio"]
        y_false_hazard_ratio = cumulative["false_hazard_ratio"]

        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Plot hazard_detection_ratio
        ax.plot(
            x_values,
            y_hazard_detection_ratio,
            label="Hazard Detection Ratio",
            color="blue",
        )

        # Plot false_hazard_ratio
        ax.plot(x_values, y_false_hazard_ratio, label="False Hazard Ratio", color="red")

        # Set labels and title
        ax.set_xlabel("Distance")
        ax.set_ylabel("Ratio")
        ax.set_title("Cumulative Results")
        ax.legend()  # Add legend

        # Display the plot
        plt.show()
        get_img_from_fig(fig).show()
