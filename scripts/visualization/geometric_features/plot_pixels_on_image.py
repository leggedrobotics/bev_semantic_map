#!/usr/bin/env python

"""
Plots pixels from point cloud projected on image on the image.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm


IMG_PATH = "/home/rschmid/RosBags/hoengg_geom_train/image"
PIXELS_PATH = "/home/rschmid/RosBags/hoengg_geom_train/np_features"

SHOW = True

if __name__ == "__main__":
    img_paths = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]
    pixel_paths = [os.path.basename(d) for d in sorted(glob.glob(PIXELS_PATH + "/*"))]

    for i in tqdm(range(len(img_paths))):
        img = torch.load(os.path.join(IMG_PATH, img_paths[i]),
                         map_location="cpu").permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_with_points = img.copy()

        pixels = np.load(os.path.join(PIXELS_PATH, pixel_paths[i]))
        # Remove pixels which are NaN
        pixels = list(map(tuple, np.argwhere(~np.isnan(pixels[:, :, 0]))))

        # print("Num pixels:", len(pixels))
        for pixel in pixels:
            cv2.circle(image_with_points, (np.round(pixel[1]).astype(int), np.round(pixel[0]).astype(int)),
                       radius=3, color=(0, 0, 255))  # Color = red

        if SHOW:
            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", image_with_points)
            cv2.waitKey(0)

        image_with_points = cv2.convertScaleAbs(image_with_points, alpha=(255.0))
        cv2.imwrite(f"/home/rschmid/{os.path.splitext(os.path.basename(img_paths[i]))[0]}.jpg", image_with_points)
