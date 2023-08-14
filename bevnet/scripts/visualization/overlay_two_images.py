#!/usr/bin/env python

"""
Overlays two images.

Author: Robin Schmid
Date: Dec 2022
"""

import os
import torch
import glob
import cv2
import numpy as np
from tqdm import tqdm

IMG_PATH1 = "/home/rschmid/Documents/final_presentation/approach/1/feat"
IMG_PATH2 = "/home/rschmid/Documents/final_presentation/approach/1/img_jpg"

SHOW = False

file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH2 + "/*"))]

for file_name in tqdm(file_names):

    # img_1 = torch.load(IMG_PATH1 + "/" + os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu'))\
    #     .permute(1, 2, 0).cpu().numpy()
    # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = cv2.imread(os.path.join(IMG_PATH1, os.path.splitext(file_name)[0] + ".jpeg"))
    img_2 = cv2.imread(os.path.join(IMG_PATH2, os.path.splitext(file_name)[0] + ".jpg"))
    # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    # img_2 = img_2.astype(np.float32)

    # img 1 is png and img 2 is jpeg, bring them to the same format
    print(img_1.shape)
    print(img_1.dtype)
    print(img_1[0, 0, :])
    print(img_2.shape)
    print(img_2.dtype)
    print(img_2[0, 0, :])
    # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # img_1 = img_1.astype(np.float32)
    # img_1 = cv2.convertScaleAbs(img_1, alpha=(255.0))

    # Overlay images
    res = cv2.addWeighted(img_1, 1, img_2, 0.8, 0)
    # res = cv2.convertScaleAbs(res, alpha=(255.0))

    if SHOW:
        cv2.imshow("res", res)
        cv2.waitKey(0)

    cv2.imwrite(f"/home/rschmid/{file_name}.jpg", res)