#!/usr/bin/env python

"""
Save .torch image as .jpg file.

Author: Robin Schmid
Date: Nov 2022
"""

import numpy as np
import os
import torch
import cv2
import glob
from tqdm import tqdm

BINARY = False
# DATA_PATH = "/home/rschmid/RosBags/output/perugia_grass/supervision_mask"
# DATA_PATH = "/home/rschmid/RosBags/arroyo_train/vehicle_mask"
DATA_PATH = "/home/rschmid/RosBags/all_train/image"

file_names = [os.path.basename(d) for d in sorted(glob.glob(DATA_PATH + "/*"))]

if __name__ == '__main__':

    for file in tqdm(file_names):

        # Load image and convert to numpy array
        if BINARY:
            img = torch.load(os.path.join(DATA_PATH, file),
                             map_location=torch.device('cpu')).cpu().numpy()

        else:
            img = torch.load(os.path.join(DATA_PATH, file),
                              map_location=torch.device('cpu')).permute(1, 2, 0).cpu().numpy()

        if BINARY:
            # Visualize bool img
            res = img.astype(np.uint8) * 255
        else:
            res = cv2.convertScaleAbs(img, alpha=(255.0))
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"/home/rschmid/RosBags/all_train/image_jpg/{os.path.splitext(os.path.basename(file))[0]}.jpg", res)
