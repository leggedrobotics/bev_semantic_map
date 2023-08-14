#!/usr/bin/env python

"""
Shows predictions from geometric features and visions features as well as points projected on the image.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_POINTS_PATH = "/home/rschmid/points"
GEOM_PRED_PATH = "/home/rschmid/geom_pred"
VISION_PRED_PATH = "/home/rschmid/vision_pred"

SAVE = True  # If true saves the images else visualizes them and allows to scroll through them

if __name__ == "__main__":
    img_points_paths = [os.path.basename(d) for d in sorted(glob.glob(IMG_POINTS_PATH + "/*"))]
    geom_pred_paths = [os.path.basename(d) for d in sorted(glob.glob(GEOM_PRED_PATH + "/*"))]
    vision_pred_paths = [os.path.basename(d) for d in sorted(glob.glob(VISION_PRED_PATH + "/*"))]

    i = 0
    while True:
        print(i)
        if i >= len(img_points_paths) or i < 0:
            exit()

        img_points = np.load(os.path.join(IMG_POINTS_PATH, img_points_paths[i]))
        geom_pred = cv2.imread(os.path.join(GEOM_PRED_PATH, geom_pred_paths[i]))
        vision_pred = cv2.imread(os.path.join(VISION_PRED_PATH, vision_pred_paths[i]))

        file = os.path.splitext(os.path.basename(img_points_paths[i]))[0]

        # Convert float32 image to unit8
        img_points = cv2.convertScaleAbs(img_points, alpha=(255.0))

        # Stack img_points and pred
        res = np.hstack((img_points, geom_pred, vision_pred))

        if SAVE:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            plt.imsave(f"/home/rschmid/res/{file}.jpg", res)
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
