#!/usr/bin/env python

"""
Creates binary labels from predictions and supervision masks for pixelwise segmentation.

Author: Robin Schmid
Date: Jan 2023
"""

import os
import sys
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

DATASET = "perugia_bin_svm_train"

LOSS_PATH = f"/home/rschmid/RosBags/{DATASET}/loss"
MASK_PATH = f"/home/rschmid/RosBags/{DATASET}/supervision_mask"
SEG_PATH = f"/home/rschmid/RosBags/{DATASET}/seg"

OUTPUT_LABEL_PATH = f"/home/rschmid/RosBags/{DATASET}/labels_seg"

np.set_printoptions(threshold=sys.maxsize)

THRESHOLD = 0.4  # Value found empirically for dataset

VISUALIZE = False

if __name__ == "__main__":

    feat_out = []
    label_out = []

    loss_paths = [os.path.basename(d) for d in sorted(glob.glob(LOSS_PATH + "/*"))]
    mask_paths = [os.path.basename(d) for d in sorted(glob.glob(MASK_PATH + "/*"))]
    seg_paths = [os.path.basename(d) for d in sorted(glob.glob(SEG_PATH + "/*"))]

    for i in tqdm(range(len(loss_paths))):

        loss = np.array(torch.load(os.path.join(LOSS_PATH, loss_paths[i]), map_location=torch.device('cpu')))  # n_seg x 1
        mask = torch.load(os.path.join(MASK_PATH, mask_paths[i]), map_location=torch.device('cpu')).numpy()  # 448 x 448
        seg = torch.load(os.path.join(SEG_PATH, seg_paths[i]), map_location=torch.device('cpu'))  # 448 x 448

        label = np.zeros_like(loss)  # Unknown = 0
        label[loss > THRESHOLD] = 2  # Unsafe = 2

        # Compute segmentation mask from loss
        for j in range(len(label)):
            seg = np.array(seg, dtype=np.float32)
            seg[seg == j] = label[j]

        seg[mask] = 1  # Safe = 1

        torch.save(seg, os.path.join(OUTPUT_LABEL_PATH, loss_paths[i]))

        if VISUALIZE:
            vis_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

            vis_img[seg == 0] = (0, 0, 0)  # Unknown
            vis_img[seg == 1] = (0, 255, 0)  # Safe
            vis_img[seg == 2] = (0, 0, 255)  # Unsafe

            cv2.imshow("img", vis_img)
            cv2.waitKey(0)
