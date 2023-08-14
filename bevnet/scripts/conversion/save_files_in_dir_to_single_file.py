#!/usr/bin/env python

"""
Saves content of files in a directory into a single file.

Author: Robin Schmid
Date: Feb 2023
"""

import os
import glob
import torch
from tqdm import tqdm

DATASET = "perugia_seg_pix"

FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/feat"
POS_FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/train_feat"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    print("Saving positive features")

    file_names = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]

    all_feat = []
    for file in tqdm(file_names):

        feat = torch.load(os.path.join(FEAT_PATH, file), map_location=DEVICE)

        all_feat.append(feat.tolist())

    torch.save(all_feat, os.path.join(POS_FEAT_PATH, "all_feat.pt"))
