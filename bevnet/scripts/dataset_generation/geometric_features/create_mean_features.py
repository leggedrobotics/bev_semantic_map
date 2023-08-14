#!/usr/bin/env python

"""
Computes torch files with the mean of the features for each segment. Excludes NaN values.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import torch

FEAT_PATH = "../../../samples/features"
SEG_PATH = "/home/rschmid/RosBags/hoengg_geom_train_vel/features/seg"

DIM_FEAT = 9

np.set_printoptions(threshold=sys.maxsize)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    print("Start")

    feat_paths = sorted(glob.glob(os.path.join(FEAT_PATH, "*")))
    seg_paths = sorted(glob.glob(os.path.join(SEG_PATH, "*")))

    for i in tqdm(range(len(feat_paths))):

        feat = np.load(feat_paths[i])
        seg = torch.load(seg_paths[i], map_location=DEVICE)

        time_stamp = feat_paths[i].split("/")[-1].split(".")[0]

        sparse_feat = np.empty((0, DIM_FEAT), np.float32)

        # Go through all segments
        for j in range(seg.max() + 1):
            # Mask features with segmentation
            mask = seg == j
            feat_masked = feat[mask]

            # Filter out nan values
            feat_masked = feat_masked[~np.isnan(feat_masked).any(axis=1)]

            # Compute mean over feat_masked
            mean = np.mean(feat_masked, axis=0)
            sparse_feat = np.append(sparse_feat, [mean], axis=0)

        # Save sparse_feat as torch tensor
        sparse_feat = torch.tensor(sparse_feat)
        torch.save(sparse_feat, f"/home/rschmid/git/wild_anomaly_detection/samples/torch_features/"
                                f"{time_stamp}.pt")
