#!/usr/bin/env python

"""
Save jpg image to tensor.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import numpy as np
import torch
from PIL import Image
import cv2

# Path to image directory
INPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask_vehicle"
OUTPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask"
items = os.listdir(INPUT_DIR)
items.sort()

print("Num items found", len(items))


if __name__ == "__main__":

    for i, item in enumerate(items):
        path = os.path.join(INPUT_DIR, item)
        if os.path.isfile(path):
            print(i)
            print(item)
            print(path)

            # img = Image.open(path)
            img = cv2.imread(path)

            # print(img)

            # For supervision mask
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.bool)

            # For RGB image
            # img = img.astype(np.float32)
            # img /= 255.0
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # show img
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            # # convert image from unit8 to float
            # img = img.astype(np.float32)
            # # convert image from BGR to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            img = torch.from_numpy(img)

            # For RGB image
            # img = img.permute(2, 0, 1)
            # print(img.shape)

            # Show image
            # img = img.permute(1, 2, 0).cpu().numpy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            torch.save(img, os.path.join(OUTPUT_DIR, (os.path.splitext(item)[0]+'.pt')))
