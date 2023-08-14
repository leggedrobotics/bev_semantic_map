#!/usr/bin/env python

"""
Fuses image of multiple cameras into one image.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

SHOW = False
SAVE = True

IMG_PATH1 = "/home/rschmid/1"
IMG_PATH2 = "/home/rschmid/2"
IMG_PATH3 = "/home/rschmid/3"

if __name__ == "__main__":
    img_paths1 = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH1 + "/*"))]
    img_paths2 = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH2 + "/*"))]
    img_paths3 = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH3 + "/*"))]

    for i, path in enumerate(tqdm(img_paths1)):

        img1 = cv2.imread(os.path.join(IMG_PATH1, img_paths1[i]))
        img2 = cv2.imread(os.path.join(IMG_PATH2, img_paths2[i]))
        img3 = cv2.imread(os.path.join(IMG_PATH3, img_paths3[i]))

        # res = np.hstack((img1, img2))
        res = np.hstack((img1, img2, img3))

        if SHOW:
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.imshow(res)
            plt.show()
            plt.close()

        if SAVE:
            cv2.imwrite(f"/home/rschmid/gif/{path}", res)
