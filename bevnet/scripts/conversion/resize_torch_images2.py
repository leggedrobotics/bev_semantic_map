#!/usr/bin/env python

"""
Resize images to a given size.

Author: Robin Schmid
Date: Feb 2023
"""

import os
import torch
import cv2
from torchvision import transforms

# Path to image directory
INPUT_DIR = "/home/rschmid/RosBags/perugia_seg_pix/labels/"
OUTPUT_DIR = "/home/rschmid/RosBags/perugia_seg_pix/labels_small/"
items = os.listdir(INPUT_DIR)
items.sort()

OUTPUT_SIZE = 224

print("Num items found", len(items))

SHOW = False

if __name__ == "__main__":

    for i, item in enumerate(items):
        path = INPUT_DIR + item
        if os.path.isfile(path):
            print(i)
            print(item)
            print(path)

            img = torch.load(path, map_location="cpu")
            img = torch.from_numpy(img)

            img = img.unsqueeze(0)

            crop = transforms.Compose(
                [
                    transforms.Resize(OUTPUT_SIZE, transforms.InterpolationMode.NEAREST),
                    transforms.CenterCrop(OUTPUT_SIZE),
                ]
            )

            img = crop(img)

            torch.save(img, OUTPUT_DIR + (os.path.splitext(item)[0]+'.pt'))

            if SHOW:
                img = img.squeeze(0)
                img = img.cpu().numpy()
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
