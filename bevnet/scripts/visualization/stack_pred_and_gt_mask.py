#!/usr/bin/env python

"""
Shows predictions as overlay as well as ground truth labels.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "/home/rschmid/RosBags/perugia_vision_long_eval/image"  # JPG or torch images
LABEL_PATH = "/home/rschmid/RosBags/perugia_vision_long_eval/gt_mask"  # Torch images
PRED_PATH = "/home/rschmid/gif"  # Jpg images
OUTPUT_PATH = "/home/rschmid/res2"

SAVE = True  # If true saves the images else visualizes them and allows to scroll through them

# EVAL_MASK = (224, 448)
EVAL_MASK = None  # If not cutting out the sky

if __name__ == "__main__":

    img_paths = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]
    label_paths = [os.path.basename(d) for d in sorted(glob.glob(LABEL_PATH + "/*"))]
    pred_paths = [os.path.basename(d) for d in sorted(glob.glob(PRED_PATH + "/*"))]

    i = 0
    while True:
        print(i)
        if i >= len(label_paths) or i < 0:
            exit()

        # img = cv2.imread(os.path.join(IMG_PATH, img_paths[i]))
        # For torch images
        img = torch.load(os.path.join(IMG_PATH, img_paths[i]),
                         map_location=torch.device('cpu')).permute(1, 2, 0).cpu().numpy()
        label = torch.load(os.path.join(LABEL_PATH, label_paths[i])).numpy()
        pred = cv2.imread(os.path.join(PRED_PATH, pred_paths[i]))

        file = os.path.splitext(os.path.basename(img_paths[i]))[0]

        overlay = img.copy()

        # For jpg images
        # overlay[~label] = [0.0, 255.0, 0.0]
        # overlay[label] = [0.0, 0.0, 255.0]

        # For torch images
        overlay[label] = [1.0, 0.0, 0.0]
        overlay[~label] = [0.0, 1.0, 0.0]

        if EVAL_MASK is not None:
            mask = np.zeros((448, 448, 3))
            mask[:EVAL_MASK[0], :EVAL_MASK[1]] = 1
            # Set seg to NaN where mask is 1 else to losses
            overlay = np.where(mask == 1, img.copy(), overlay)

        res = cv2.addWeighted(overlay, 0.2, img, 0.8, 0.0)

        # For torch images
        res = cv2.convertScaleAbs(res, alpha=(255.0))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        res = np.hstack((pred, res))

        if SAVE:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            plt.imsave(f"{OUTPUT_PATH}/{file}.jpg", res)
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
