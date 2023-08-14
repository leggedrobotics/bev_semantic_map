#!/usr/bin/env python

"""
Resize images to a given size.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import torch
import cv2
from torchvision import transforms

# Path to image directory
INPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask_jpg"
OUTPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask_jpg_448"
SHOW = False

items = os.listdir(INPUT_DIR)
items.sort()

print("Num items found", len(items))

if __name__ == "__main__":

    for i, item in enumerate(items):
        path = os.path.join(INPUT_DIR, item)
        if os.path.isfile(path):
            # print(i)
            # print(item)
            print(path)

            img = cv2.imread(path)
            # Change type to bool
            # img = torch.load(path, map_location=device)

            # Resize image
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
            # img = cv2.copyMakeBorder(img, 0, 4, 0, 0, cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.splitext(item)[0]+'.jpg'), img)

            if SHOW:
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
