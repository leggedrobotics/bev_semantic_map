#!/usr/bin/env python

"""
Creates a torch file with all features stacked which are inside of the provided binary mask.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import glob
import torch
import numpy as np
from tqdm import tqdm

DATASET = "arroyo_train"

FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/features/all_feat"
SEG_PATH = f"/home/rschmid/RosBags/{DATASET}/features/seg"
MASK_PATH = f"/home/rschmid/RosBags/{DATASET}/supervision_mask"
POS_FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/features/pos_feat"

DEVICE = torch.device('cpu')  # Usually faster than GPU

if __name__ == "__main__":
    print("Saving positive features")

    file_names = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]
    p_feat = []
    for file in tqdm(file_names):

        with torch.no_grad():
            feat = torch.load(os.path.join(FEAT_PATH, file), map_location=DEVICE)
            seg = torch.load(os.path.join(SEG_PATH, file), map_location=DEVICE)
            mask = torch.load(os.path.join(MASK_PATH, file), map_location=DEVICE)

            # Iterate through all segments of the segmentation
            # print("Num segments: ", seg.max().item())
            for s in torch.unique(seg):
                m = mask[seg == s]
                pos = m.sum()  # Inside
                neg = (~m).sum()  # Outside

                # Check if more inside features than outside features
                if pos > neg:
                    # Check if the vector contains any NaN values
                    if not np.isnan(feat[s.item()].cpu()).any():
                        p_feat.append(feat[s.item()])

    p_feat = torch.stack(p_feat, dim=1).T

    torch.save(p_feat, os.path.join(POS_FEAT_PATH, "pos_feat.pt"))
