#!/usr/bin/env python

"""
Overlays mask of vehicle on image.

Author: Robin Schmid
Date: Dec 2022
"""

import os
import torch
import glob
import cv2
import numpy as np
from tqdm import tqdm

VEHICLE_MASK = "/home/rschmid/RosBags/helendale_train/vehicle_mask/vehicle_mask.jpg"
IMG_PATH = "/home/rschmid/1"

OUTPUT_PATH = "/home/rschmid/3"

# VEHICLE_MASK = "/home/rschmid/RosBags/arroyo_train/vehicle_mask/vehicle_mask.jpg"
# SUPERVISION_MASK = "/home/rschmid/RosBags/arroyo_train/supervision_mask_jpg_448"
#
# OUTPUT_PATH = "/home/rschmid/RosBags/arroyo_train/supervision_mask_vehicle"

SHOW = False

file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]

vehicle_mask = cv2.imread(VEHICLE_MASK)

for file_name in tqdm(file_names):

    img = cv2.imread(os.path.join(IMG_PATH, file_name))
    # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # img_2 = cv2.imread(os.path.join(IMG_PATH2, file_name))
    # # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    # img_2 = img_2.astype(np.float32)

    res = cv2.bitwise_and(img, vehicle_mask)

    # Overlay images
    # res = cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0)
    # res = cv2.convertScaleAbs(res, alpha=(255.0))

    if SHOW:
        cv2.imshow("res", res)
        cv2.waitKey(0)

    cv2.imwrite(os.path.join(OUTPUT_PATH, f"{file_name}"), res)
