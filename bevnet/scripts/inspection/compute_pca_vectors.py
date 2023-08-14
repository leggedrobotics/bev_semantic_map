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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


FEAT_PATH = "/home/rschmid/RosBags/sa_walk_geom/features/all_feat"
SEG_PATH = "/home/rschmid/RosBags/sa_walk_geom/features/seg"
MASK_PATH = "/home/rschmid/RosBags/sa_walk_geom/supervision_mask"

MAX_IT = 100

FEAT_DIM = 5

START_IDX = 700
STOP_IDX = 730

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]

    p_feat = np.empty((0, FEAT_DIM), np.float32)
    u_feat = np.empty((0, FEAT_DIM), np.float32)

    it = 0

    for i in tqdm(range(START_IDX, STOP_IDX)):
        file = file_names[i]

        if it > MAX_IT:
            break

        feat = torch.load(os.path.join(FEAT_PATH, file), map_location=torch.device('cpu'))
        seg = torch.load(os.path.join(SEG_PATH, file), map_location=torch.device('cpu'))
        mask = torch.load(os.path.join(MASK_PATH, file), map_location=torch.device('cpu'))

        # Normalize normal vector
        n_xyz = feat.numpy()[:, :3]
        n_xyz = n_xyz / np.linalg.norm(n_xyz, axis=1)[:, np.newaxis]

        # Horizontal normal vector component
        n_h = np.linalg.norm(n_xyz[:, :2], axis=1)
        n_z = n_xyz[:, 2]

        # Other features
        curv = feat.numpy()[:, 3]
        princ_comp = feat.numpy()[:, 7:9]

        # Concatenate n_h, n_z, curv and princ_comp
        feat = np.concatenate((n_h[:, np.newaxis], n_z[:, np.newaxis], curv[:, np.newaxis], princ_comp), axis=1)

        # print(feat.shape)

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
                    p_feat_curr = feat[s.item()]
                    p_feat = np.append(p_feat, [p_feat_curr], axis=0)
            else:
                # Check if the vector contains any NaN values
                if not np.isnan(feat[s.item()]).any():
                    u_feat_curr = feat[s.item()]
                    u_feat = np.append(u_feat, [u_feat_curr], axis=0)

        # print("p_feat", np.mean(p_feat, axis=0))
        # print("u_feat", np.mean(u_feat, axis=0))
        # print("\n")

        it += 1

    # Concatenate p_feat and u_feat
    feat = np.concatenate((p_feat, u_feat), axis=0)

    # Create vector with zeros of length p_feat and ones of length u_feat
    y = np.concatenate((np.zeros(p_feat.shape[0]), np.ones(u_feat.shape[0])), axis=0)

    pca = PCA(2)  # Project to 2 dimensions
    projected = pca.fit_transform(feat)

    # print(p_feat.shape)
    # print(projected.shape)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.scatter(projected[:, 0], projected[:, 1], s=10,
                c=y, alpha=0.5,
                cmap=plt.cm.get_cmap('cool'))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('PCA')
    plt.colorbar()

    plt.show()
