#!/usr/bin/env python

"""
Save .torch image as .jpg file.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import numpy as np
import cv2
import glob
from tqdm import tqdm


DATA_PATH = "/home/rschmid/git/wild_anomaly_detection/samples/image_with_points/"

file_names = [os.path.basename(d) for d in sorted(glob.glob(DATA_PATH + "/*"))]

if __name__ == '__main__':

    for file in tqdm(file_names):

        # Load image and convert to numpy array
        img = np.load(os.path.join(DATA_PATH, file))

        # Convert from float to uint8 image and change color encoding
        res = cv2.convertScaleAbs(img, alpha=(255.0))
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        #
        # cv2.imshow("res", res)
        # cv2.waitKey(0)

        cv2.imwrite(f"{os.path.splitext(os.path.basename(file))[0]}.jpg", res)
