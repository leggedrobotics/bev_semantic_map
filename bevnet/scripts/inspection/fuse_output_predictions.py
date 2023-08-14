#!/usr/bin/env python

"""
Fuse predictions from two different models.

Author: Robin Schmid
Date: Dec 2022
"""

import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH_1 = "/home/rschmid/geom_pred"
PRED_PATH_2 = "/home/rschmid/vision_pred"
IMG_PATH = "/home/rschmid/RosBags/sa_eval_geom/image"
SEG_PATH = "/home/rschmid/RosBags/sa_eval_geom/features/seg"

OUTPUT_PATH_TORCH = "/home/rschmid/res_torch"
OUTPUT_PATH_IMG = "/home/rschmid/res"

SAVE_TORCH = True
SAVE_IMG = False  # If true saves the images else visualizes them and allows to scroll through them

EVAL_MASK = (224, 448)
THRESHOLD = 0.1  # 0.3, 0.5, 0.3  # Threshold for safe region;
# lower: more is safe / more is green; higher: less is safe / more is red
UNKNOWN_INTERVAL = [-0.1, 0.1]  # Interval of predictions of unknown class
OVERLAY_ALPHA = 0.35  # Overlay of traversability mask

if __name__ == "__main__":

    pred_paths_1 = [os.path.basename(d) for d in sorted(glob.glob(PRED_PATH_1 + "/*"))]
    pred_paths_2 = [os.path.basename(d) for d in sorted(glob.glob(PRED_PATH_2 + "/*"))]
    img_paths = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]
    seg_paths = [os.path.basename(d) for d in sorted(glob.glob(SEG_PATH + "/*"))]

    i = 0
    while True:
        print(i)
        if i >= len(pred_paths_1) or i < 0:
            exit()

        pred_1 = torch.load(os.path.join(PRED_PATH_1, pred_paths_1[i]), map_location="cpu")
        pred_2 = torch.load(os.path.join(PRED_PATH_2, pred_paths_2[i]), map_location="cpu")
        img = torch.load(os.path.join(IMG_PATH, img_paths[i]), map_location="cpu").permute(1, 2, 0).cpu().numpy()
        # seg = torch.load(os.path.join(SEG_PATH, seg_paths[i]))

        file = os.path.splitext(os.path.basename(img_paths[i]))[0]

        # Fuse the predictions
        seg = pred_1 * pred_2

        if SAVE_TORCH:
            torch.save(seg, os.path.join(OUTPUT_PATH_TORCH, file + ".pt"))

        # Split into safe and unsafe regions based on threshold
        # Safe regions
        overlay = img.copy()
        overlay[:] = [0.0, 1.0, 0.0] - np.multiply.outer(
            seg * (1 / THRESHOLD),
            [0.0, 1.0, 0.0],
        )
        overlay[seg >= THRESHOLD] = 0

        # Unsafe regions
        seg[seg < THRESHOLD] = 0
        overlay[:] += np.multiply.outer(
            seg * (1 / (1 - THRESHOLD)),
            [1.0, 0.0, 0.0],
        )

        # Unkonwn regions
        overlay[np.where((seg > THRESHOLD + UNKNOWN_INTERVAL[0]) &
                         (seg < THRESHOLD + UNKNOWN_INTERVAL[1]), 0, 1) == 0] = [0.0, 0.0, 1.0]

        # Only plot generate overlay for non-NaN values
        overlay[np.isnan(seg)] = img[np.isnan(seg)]

        # Overlay safe and unsafe regions with image
        res = cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1 - OVERLAY_ALPHA, 0.0)

        if EVAL_MASK is not None:
            mask = np.zeros((448, 448, 3))
            mask[:EVAL_MASK[0], :EVAL_MASK[1]] = 1
            # Set seg to NaN where mask is 1 else to losses
            res = np.where(mask == 1, img.copy(), res)

        # Convert to BGR for opencv
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        if SAVE_IMG:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            plt.imsave(f"{OUTPUT_PATH_IMG}/{file}.jpg", res)
            i += 1
        else:
            cv2.imshow("res", res)
            cv2.waitKey(0)

            while True:
                key = cv2.waitKey(0)
                if key is ord("e"):
                    i += 1
                    break
                elif key is ord("q"):
                    i -= 1
                    break
                else:
                    exit()
