#!/usr/bin/env python

"""
Concatenate tensors from multiple files into one tensor.

Author: Robin Schmid
Date: Dec 2022
"""

import os
import glob
import torch
import numpy as np
from tqdm import tqdm

CONVERT_FEAT = True

FEAT_PATH_1 = "/home/rschmid/RosBags/sa_walk_geom/features/all_feat"
FEAT_PATH_2 = "/home/rschmid/RosBags/sa_walk_vision/features/all_feat"
OUTPUT_PATH = "/home/rschmid/RosBags/sa_walk_fused/features/all_feat"


def convert_feat(feat):
    # Normalize normal vector
    n_xyz = feat[:, :3]
    n_xyz = n_xyz / np.linalg.norm(n_xyz, axis=1)[:, np.newaxis]

    # Horizontal normal vector component
    n_h = np.linalg.norm(n_xyz[:, :2], axis=1)
    n_z = n_xyz[:, 2]

    # Angle between n_h and n_z for each element
    n_angle = np.arctan2(n_z, n_h)

    # Other features, curvature and principal components
    curv = feat[:, 3]
    princ_comp = feat[:, 7:9]

    # Concatenate n_h, n_z, curv and princ_comp
    # feat = np.concatenate((n_h[:, np.newaxis], n_z[:, np.newaxis], curv[:, np.newaxis], princ_comp), axis=1)
    # feat = np.concatenate((n_h[:, np.newaxis], n_z[:, np.newaxis],
    # n_angle[:, np.newaxis], curv[:, np.newaxis], princ_comp), axis=1)
    feat = np.concatenate((n_angle[:, np.newaxis], curv[:, np.newaxis], princ_comp), axis=1)

    return feat


if __name__ == "__main__":
    print("Saving positive features")

    file_names = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH_1 + "/*"))]

    for file in tqdm(file_names):

        feat_1 = torch.load(os.path.join(FEAT_PATH_1, file), map_location=torch.device('cpu'))

        if CONVERT_FEAT:
            feat_1 = convert_feat(feat_1)

        feat_1 = torch.from_numpy(feat_1)

        feat_2 = torch.load(os.path.join(FEAT_PATH_2, file), map_location=torch.device('cpu'))

        # print(feat_1.shape)
        # print(feat_2.shape)

        # Concatenate both features with torch
        feat = torch.cat((feat_1, feat_2), dim=1)

        # print(feat.shape)
        torch.save(feat, os.path.join(OUTPUT_PATH, file))
