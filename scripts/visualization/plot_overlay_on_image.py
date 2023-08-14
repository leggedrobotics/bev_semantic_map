#!/usr/bin/env python

"""
Plot mask on the image as an overlay.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import torch
import cv2
import glob
from tqdm import tqdm

# IMG_PATH = "/home/rschmid/RosBags/hoengg_vision_train/image"
# MASK_PATH = "/home/rschmid/RosBags/hoengg_vision_train/supervision_mask"

IMG_PATH = "/home/rschmid/RosBags/helendale_train/image"
MASK_PATH = "/home/rschmid/RosBags/helendale_train/supervision_mask"

# IMG_PATH = "/home/rschmid/RosBags/perugia_100_eval/image"
# MASK_PATH = "/home/rschmid/RosBags/perugia_100_eval/labels"

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]

    for file_name in tqdm(file_names):
        img = torch.load(os.path.join(IMG_PATH, file_name), map_location=torch.device('cpu')).permute(1, 2, 0).cpu().numpy()
        # print(img.shape)
        # print(img.dtype)
        # print(img)
        # img = cv2.imread(os.path.join(IMG_PATH, file_name))
        mask = torch.load(MASK_PATH + "/" + os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu')).cpu().numpy()
        # print(mask.shape)
        # print(mask.dtype)

        overlay = img.copy()
        overlay[mask] = [0.0, 1.0, 0.0]  # Red color
        # overlay[mask] = [0.0, 0.0, 255.0]

        res = cv2.addWeighted(overlay, 0.2, img, 0.8, 0.0)

        # For torch images
        res = cv2.convertScaleAbs(res, alpha=(255.0))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        # cv2.imshow("overlay", res)
        # cv2.waitKey(0)

        cv2.imwrite(f"/home/rschmid/overlay/{os.path.splitext(os.path.basename(file_name))[0]}.jpg", res)
