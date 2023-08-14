#!/usr/bin/env python

"""
Save .torch image as .jpg file.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import torch
import cv2
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image


DATA_PATH = "/home/rschmid/RosBags/perugia_bin_svm_train/label_pix"

VISUALIZE = False

file_names = [os.path.basename(d) for d in sorted(glob.glob(DATA_PATH + "/*"))]

if __name__ == '__main__':

    for file in tqdm(file_names):

        # Load image and convert to numpy array
        img = torch.load(os.path.join(DATA_PATH, file), map_location=torch.device('cpu'))

        if VISUALIZE:
            vis_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            vis_img[img == 0] = (0, 0, 0)  # Unknown
            vis_img[img == 1] = (0, 255, 0)  # Safe
            vis_img[img == 2] = (0, 0, 255)  # Unsafe

            cv2.imshow("img", vis_img)
            cv2.waitKey(0)

        img = Image.fromarray(img)
        img.save(f"/home/rschmid/{os.path.splitext(os.path.basename(file))[0]}.gif")
