#!/usr/bin/env python

"""
Computes evaluation metrics for given predictions.

Author: Robin Schmid
Date: Apr 2023
"""

import os
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

IND_PLOTS = False  # Individual ROC plots per prediction

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Max Youdens Index (adapted from Lorenz Wellhausen)
def compute_max_youdens_index(fpr, tpr, thr):
    max_val = 0
    max_ind = None
    for i, (fp, tp) in enumerate(zip(fpr, tpr)):
        cur_val = tp - fp
        if cur_val > max_val:
            max_val = cur_val
            max_ind = i
    return fpr[max_ind], tpr[max_ind], max_val, thr[max_ind]


def compute_iou(pred, gt, thr):
    pred = pred > thr
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_accuracy(pred, gt, thr):
    pred = pred > thr
    accuracy = np.sum(pred == gt) / np.prod(gt.shape)
    return accuracy


def compute_dice_score(pred, gt, thr):
    pred = pred > thr
    intersection = np.sum(np.logical_and(pred, gt))
    dice_score = 2.0 * intersection / (np.sum(pred) + np.sum(gt))
    return dice_score


def compute_tpr_at_x_fpr(fpr, tpr, fpr_thres):
    crit_ind = 0
    for ind, (fp, tp) in enumerate(zip(fpr, tpr)):
        if fp > fpr_thres:
            crit_ind = ind
            break
    if crit_ind == 0:
        fp_lower = 0.0
        tp_lower = 0.0
    else:
        fp_lower = fpr[crit_ind-1]
        tp_lower = tpr[crit_ind-1]
    fp_upper = fpr[crit_ind]
    tp_upper = tpr[crit_ind]

    tp_crit = tp_lower + (tp_upper-tp_lower)*(fpr_thres-fp_lower)/(fp_upper-fp_lower)
    return tp_crit


def compute_evaluation(gt_path, pred_path, fig_path, model_name=None, threshold=None):
    print("Computing ROC plot...")

    file_names = [os.path.basename(d) for d in sorted(glob.glob(pred_path + "/*"))]

    gt_all = np.array([])
    pred_all = np.array([])

    gt_paths = sorted(glob.glob(gt_path + "/*"))
    pred_paths = sorted(glob.glob(pred_path + "/*"))

    for i, file_name in enumerate(tqdm(file_names)):

        gt = torch.load(gt_paths[i], map_location=DEVICE)
        pred = torch.load(pred_paths[i], map_location=DEVICE)[0]

        gt[gt == -1] = np.nan
        pred[pred == -1] = np.nan

        gt = gt.flatten()
        pred = pred.flatten()

        # print(gt.shape, pred.shape)

        # Remove elements where predictions are NaN values
        # gt = gt[~np.isnan(pred)]
        # pred = pred[~np.isnan(pred)]

        # gt = gt[~np.isnan(gt)]
        # pred = pred[~np.isnan(gt)]

        # Create a mask where either gt or pred is NaN
        mask = ~np.isnan(gt) & ~np.isnan(pred)
        
        # Apply the mask to both gt and pred to remove NaN values
        gt = gt[mask]
        pred = pred[mask]

        # print(gt.shape, pred.shape)

        # print(gt.shape, pred.shape)

        gt_all = np.append(gt_all, gt)
        pred_all = np.append(pred_all, pred)

        # Individual ROC plots per image
        if IND_PLOTS:
            try:
                auc = roc_auc_score(gt, pred)
                fpr, tpr, thr = roc_curve(gt, pred)
                maxY = compute_max_youdens_index(fpr, tpr, thr)

                fig, ax = plt.subplots()
                # ax.title.set_text("ROC Curve")
                ax.set(xlabel='FPR', ylabel='TPR',
                       title="AUC: {:.2f}%, maxYoudens: {:.2f}".format(100. * auc, maxY))
                ax.plot(fpr, tpr)
                ax.grid()
                plt.gca().set_aspect("equal", adjustable="box")
                fig.savefig(f"{fig_path}/ind/{file_name}"[:-3] + ".png")
                plt.close(fig)
            except ValueError:
                print("ValueError: ROC plot could not be computed for " + file_name)
                pass

    auc_all = roc_auc_score(gt_all, pred_all)
    fpr_all, tpr_all, thr_all = roc_curve(gt_all, pred_all)
    fprY_all, tprY_all, maxY_all, maxThr_all = compute_max_youdens_index(fpr_all, tpr_all, thr_all)

    tprat5fpr_all = compute_tpr_at_x_fpr(fpr_all, tpr_all, 0.05)

    if threshold is None:
        threshold = maxThr_all
        print("Using the best possible threshold")

    # iou = compute_iou(pred_all, gt_all, threshold)
    # acc = compute_accuracy(pred_all, gt_all, threshold)
    # dice = compute_dice_score(pred_all, gt_all, threshold)

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.suptitle(f"{model_name} \n", fontsize=5)
    ax.set(xlabel='FPR', ylabel='TPR',
                   title="AUC: {:.2f}%, TPR at 5% FPR: {:.2f}, max Youdens: {:.2f}, max Thr: {:.2f}"
                   .format(100. * auc_all, maxY_all, tprat5fpr_all, maxThr_all))
                           # ax.set(xlabel='FPR', ylabel='TPR',
    #                title="AUC: {:.2f}%, TPR at 5% FPR: {:.2f} \n "
    #                      "max Youdens: {:.2f}, max Thr: {:.2f} \n"
    #                      "Acc: {:.2f}%, IoU: {:.2f}%, Dice: {:.2f}% \n"
    #                .format(100. * auc_all, maxY_all, tprat5fpr_all, maxThr_all,
    #                        100. * acc, 100. * iou, 100. * dice))

    ax.plot(fpr_all, tpr_all)

    # Draw line at max Youdens index
    ax.plot([fprY_all, fprY_all], [fprY_all, tprY_all], 'r-')
    ax.plot([0.05, 0.05], [0, tprat5fpr_all], 'g-')

    # Plot 45 degree line
    x = np.linspace(0, 1)
    ax.plot(x, x, "--", color="black")

    ax.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{fig_path}/roc_plot.png")
    plt.show()
