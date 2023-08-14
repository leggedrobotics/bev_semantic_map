#!/usr/bin/env python

"""
Creates pixelwise feature labels from the segmentation mask.
Data saved needs a lot of memory!

Author: Robin Schmid
Date: Feb 2023
"""

import os
import sys
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

DATASET = "perugia_bin_svm_train"

FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/feat"
SEG_PATH = f"/home/rschmid/RosBags/{DATASET}/seg"

OUTPUT_LABEL_PATH = f"/home/rschmid/RosBags/{DATASET}/feat_pix"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":

    feat_out = []
    label_out = []

    feat_paths = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]
    seg_paths = [os.path.basename(d) for d in sorted(glob.glob(SEG_PATH + "/*"))]

    for i in tqdm(range(len(feat_paths))):

        feat = np.array(torch.load(os.path.join(FEAT_PATH, feat_paths[i]), map_location=torch.device(DEVICE)))  # n_seg x 90
        seg = torch.load(os.path.join(SEG_PATH, seg_paths[i]), map_location=torch.device(DEVICE))  # 448 x 448

        # Compute segmentation mask from loss
        for j in range(seg.max()):
            seg = np.array(seg, dtype=np.float32)

            # Expand seg by 90 dimension
            seg_exp = np.zeros((seg.shape[0], seg.shape[1], feat.shape[1]))
            seg_exp[seg == j] = feat[j]

        torch.save(seg_exp, os.path.join(OUTPUT_LABEL_PATH, feat_paths[i]))
