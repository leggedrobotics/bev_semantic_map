#!/usr/bin/env python

"""
Creates binary labels from predictions and supervision masks.

Author: Robin Schmid
Date: Jan 2023
"""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

DATASET = "hoengg_bin_train"

IMG_PATH = f"/home/rschmid/RosBags/{DATASET}/image"
FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/features_dino"
LOSS_PATH = f"/home/rschmid/RosBags/{DATASET}/output_loss"
MASK_PATH = f"/home/rschmid/RosBags/{DATASET}/supervision_mask"
SEG_PATH = f"/home/rschmid/RosBags/{DATASET}/features/seg"

OUTPUT_FEAT_PATH = f"/home/rschmid/RosBags/{DATASET}/train_feat2"
OUTPUT_LABEL_PATH = f"/home/rschmid/RosBags/{DATASET}/train_label2"
OUTPUT_LABEL_IMG_PATH = f"/home/rschmid/RosBags/{DATASET}/label_img3"
OUTPUT_OVERLAY_IMG_PATH = f"/home/rschmid/RosBags/{DATASET}/overlay_img3"

THRESHOLD = 0.5

VISUALIZE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    feat_out = []
    label_out = []

    feat_paths = [os.path.basename(d) for d in sorted(glob.glob(FEAT_PATH + "/*"))]
    loss_paths = [os.path.basename(d) for d in sorted(glob.glob(LOSS_PATH + "/*"))]
    mask_paths = [os.path.basename(d) for d in sorted(glob.glob(MASK_PATH + "/*"))]
    seg_paths = [os.path.basename(d) for d in sorted(glob.glob(SEG_PATH + "/*"))]

    for i in tqdm(range(len(feat_paths))):
        img = torch.load(os.path.join(IMG_PATH, feat_paths[i]), map_location=torch.device('cpu'))\
            .permute(1, 2, 0).cpu().numpy()  # 448 x 448 x 3
        feat = torch.load(os.path.join(FEAT_PATH, feat_paths[i]), map_location=torch.device('cpu'))  # n_seg x 384
        loss = np.array(torch.load(os.path.join(LOSS_PATH, loss_paths[i]), map_location=torch.device('cpu')))  # n_seg x 1
        mask = torch.load(os.path.join(MASK_PATH, mask_paths[i]), map_location=torch.device('cpu')).numpy()  # 448 x 448
        seg = torch.load(os.path.join(SEG_PATH, seg_paths[i]), map_location=torch.device('cpu'))  # 448 x 448

        label = np.zeros_like(loss)  # Unknown = 0

        label[loss > THRESHOLD] = 2  # Unsafe = 2

        # If region is both safe and unsafe set to safe
        # for s in torch.unique(seg):
        #     m = mask[seg == s]
        #     pos = m.sum()  # Inside
        #     neg = (~m).sum()  # Outside
        #
        #     # Heuristic to remove some negative labels
        #     # if s.item() > 800:
        #     #     label[s.item()] = 0
        #
        #     # Check if more inside features than outside features
        #     if pos > neg:
        #         # Check if the vector contains any NaN values
        #         if not np.isnan(feat[s.item()]).any():
        #             label[s.item()] = 1  # Save = 1

        # Remove all 0 elements in label
        label_non_zero = label[label != 0]

        # Fix labels such that safe is 0 and unsafe is 1
        label_non_zero[label_non_zero == 1] = 0
        label_non_zero[label_non_zero == 2] = 1

        # Remove all elements in feat where label is 0
        feat_non_zero = feat[label != 0]

        # feat_out.extend(feat_non_zero.tolist())
        label_out.extend(label_non_zero.tolist())

        if VISUALIZE:
            vis_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)

            for j in range(len(label)):
                seg = np.array(seg, dtype=np.float32)
                seg[seg == j] = label[j]

            vis_img[seg == 0] = (0, 0, 0)  # Unknown
            vis_img[seg == 1] = (0, 1, 0)  # Safe
            vis_img[seg == 2] = (0, 0, 1)  # Unsafe
            res = cv2.convertScaleAbs(vis_img, alpha=(255.0))

            cv2.imwrite(os.path.splitext(os.path.join(OUTPUT_LABEL_IMG_PATH, loss_paths[i]))[0] + ".png", res)
            # cv2.imshow("img", vis_img)
            # cv2.waitKey(0)

            # Generate overlay
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img, 0.7, vis_img, 0.3, 0)
            res = cv2.convertScaleAbs(overlay, alpha=(255.0))
            cv2.imwrite(os.path.splitext(os.path.join(OUTPUT_OVERLAY_IMG_PATH, loss_paths[i]))[0] + ".png", res)

    # torch.save(feat_out, os.path.join(OUTPUT_FEAT_PATH, "train_feat.pt"))
    # torch.save(label_out, os.path.join(OUTPUT_LABEL_PATH, "train_label.pt"))
