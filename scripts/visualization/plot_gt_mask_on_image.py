#!/usr/bin/env python

"""
Plot label on the image as an overlay. Labels are binary from segments.ai.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import torch
import cv2
import glob
from tqdm import tqdm

# IMG_PATH = "/home/rschmid/RosBags/hoengg_vision_eval/image"
# MASK_PATH = "/home/rschmid/RosBags/hoengg_vision_eval/gt_mask"

IMG_PATH = "/home/rschmid/RosBags/sa_vision_eval/image"
MASK_PATH = "/home/rschmid/RosBags/sa_vision_eval/gt_mask"

# IMG_PATH = "/home/rschmid/RosBags/perugia_100_eval/image"
# MASK_PATH = "/home/rschmid/RosBags/perugia_100_eval/labels"

VISUALIZE = False

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]

    for file_name in tqdm(file_names):
        img = torch.load(os.path.join(IMG_PATH, file_name), map_location=torch.device('cpu')).permute(1, 2, 0).cpu().numpy()
        # img = cv2.imread(os.path.join(IMG_PATH, file_name))
        mask = torch.load(MASK_PATH + "/" + os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu')).cpu().numpy()

        overlay = img.copy()
        overlay[~mask] = [0.0, 1.0, 0.0]
        overlay[mask] = [1.0, 0.0, 0.0]

        res = cv2.addWeighted(overlay, 0.2, img, 0.8, 0.0)

        # For torch images
        res = cv2.convertScaleAbs(res, alpha=(255.0))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        if VISUALIZE:
            cv2.imshow("overlay", res)
            cv2.waitKey(0)

        cv2.imwrite(f"/home/rschmid/overlay/{os.path.splitext(os.path.basename(file_name))[0]}.jpg", res)
