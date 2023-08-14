#!/usr/bin/env python

"""
Shows mean features for positive and unlabeled features

Author: Robin Schmid
Date: Dec 2022
"""

import os
import glob
import torch
import numpy as np
from tqdm import tqdm


FEAT_PATH = "/home/rschmid/RosBags/sa_walk_geom/features/all_feat"
SEG_PATH = "/home/rschmid/RosBags/sa_walk_geom/features/seg"
MASK_PATH = "/home/rschmid/RosBags/sa_walk_geom/supervision_mask"

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]

    p_feat = np.empty((0, 9), np.float32)
    u_feat = np.empty((0, 9), np.float32)

    for file in tqdm(file_names):

        feat = torch.load(os.path.join(FEAT_PATH, file), map_location=torch.device('cpu'))
        seg = torch.load(os.path.join(SEG_PATH, file), map_location=torch.device('cpu'))
        mask = torch.load(os.path.join(MASK_PATH, file), map_location=torch.device('cpu'))

        # Iterate through all segments of the segmentation and save positive and unlabeled features
        # print("Compute positive and unlabeled features")
        for s in torch.unique(seg):
            m = mask[seg == s]
            pos = m.sum()  # Inside
            neg = (~m).sum()  # Outside

            # Check if more inside features than outside features
            if pos > neg:
                # Check if the vector contains any NaN values
                if not np.isnan(feat[s.item()]).any():
                    p_feat = np.append(p_feat, [feat[s.item()].numpy()], axis=0)
            else:
                # Check if the vector contains any NaN values
                if not np.isnan(feat[s.item()]).any():
                    u_feat = np.append(u_feat, [feat[s.item()].numpy()], axis=0)

        print("p_feat", np.mean(p_feat, axis=0))
        print("u_feat", np.mean(u_feat, axis=0))
        print("\n")

    # torch.save(p_feat, os.path.join(POS_FEAT_PATH, "pos_feat.pt"))
