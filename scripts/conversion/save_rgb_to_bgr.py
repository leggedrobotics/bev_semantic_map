#!/usr/bin/env python

"""
Save rgb to bgr of vice versa.

Author: Robin Schmid
Date: Mar 2023
"""

import os
import cv2
import glob
from tqdm import tqdm


DATA_PATH = "/home/rschmid/Documents/final_presentation/approach/2/feat"

file_names = [os.path.basename(d) for d in sorted(glob.glob(DATA_PATH + "/*"))]

if __name__ == '__main__':

    for file in tqdm(file_names):

        # Load image and convert to numpy array
        img = cv2.imread(os.path.join(DATA_PATH, file))

        res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.imshow("res", res)
        # cv2.waitKey(0)

        cv2.imwrite(f"/home/rschmid/{os.path.splitext(os.path.basename(file))[0]}.jpg", res)
