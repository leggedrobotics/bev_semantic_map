#!/usr/bin/env python

"""
Refines the loss for a finer segmentation for training masks.

Author: Robin Schmid
Date: Jan 2023
"""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

INPUT_LOSS_PATH = "/home/rschmid/RosBags/perugia_bin_fine/loss"
INPUT_SEG_PATH = "/home/rschmid/RosBags/perugia_bin_fine/old_seg"

NEW_SEG_PATH = "/home/rschmid/RosBags/perugia_bin_fine/features/seg"
OUTPUT_LOSS_PATH = "/home/rschmid/RosBags/perugia_bin_fine/output_loss"

VISUALIZE = False

if __name__ == "__main__":

    feat_out = []
    label_out = []

    input_loss_paths = [os.path.basename(d) for d in sorted(glob.glob(INPUT_LOSS_PATH + "/*"))]
    input_seg_paths = [os.path.basename(d) for d in sorted(glob.glob(INPUT_SEG_PATH + "/*"))]
    new_seg_paths = [os.path.basename(d) for d in sorted(glob.glob(NEW_SEG_PATH + "/*"))]
    output_loss_paths = [os.path.basename(d) for d in sorted(glob.glob(OUTPUT_LOSS_PATH + "/*"))]

    for i in tqdm(range(len(input_loss_paths))):
        loss = np.array(torch.load(os.path.join(INPUT_LOSS_PATH, input_loss_paths[i]), map_location=torch.device('cpu')))  # n_seg x 1
        seg = torch.load(os.path.join(INPUT_SEG_PATH, input_seg_paths[i]), map_location=torch.device('cpu'))  # 448 x 448
        new_seg = torch.load(os.path.join(NEW_SEG_PATH, new_seg_paths[i]), map_location=torch.device('cpu'))  # 448 x 448

        for j in range(len(torch.unique(seg))):
            seg = np.array(seg, dtype=np.float32)
            seg[seg == j] = loss[j]

        # Compute new loss as mean of old loss withing segments
        output_loss = [np.mean(seg[new_seg == s]) for s in torch.unique(new_seg)]

        # print(len(output_loss))
        # print(len(torch.unique(new_seg)))

        # print(output_loss)
        seg_out = np.array(new_seg, dtype=np.float32)
        for idx, s in enumerate(torch.unique(new_seg)):
            seg_out[seg_out == s.item()] = output_loss[idx]

        # Visualize seg_out
        if VISUALIZE:
            res = np.hstack((seg, seg_out))
            cv2.imshow("res", res)
            cv2.waitKey(0)

        # Save output loss
        torch.save(output_loss, os.path.join(OUTPUT_LOSS_PATH, new_seg_paths[i]))

    # torch.save(feat_out, os.path.join(OUTPUT_FEAT_PATH, "train_feat.pt"))
    # torch.save(label_out, os.path.join(OUTPUT_LABEL_PATH, "train_label.pt"))
