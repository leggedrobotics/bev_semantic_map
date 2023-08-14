#!/usr/bin/env python

"""
Resize images to a given size.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import torch
import cv2
import torchvision

# Path to image directory
INPUT_DIR = "/home/rschmid/RosBags/hoengg_vision_low_res_eval/gt_mask"
OUTPUT_DIR = "/home/rschmid/RosBags/hoengg_vision_low_res_eval/gt_mask_small"
items = os.listdir(INPUT_DIR)
items.sort()

print("Num items found", len(items))

SHOW = False

if __name__ == "__main__":

    for i, item in enumerate(items):
        path = os.path.join(INPUT_DIR, item)
        if os.path.isfile(path):
            print(i)
            print(item)
            print(path)

            img = torch.from_numpy(torch.load(path, map_location="cpu").cpu().numpy()).unsqueeze(0)

            img = torchvision.transforms.Resize((224, 224),
                                          torchvision.transforms.InterpolationMode.BILINEAR)(img).squeeze()

            torch.save(img, os.path.join(OUTPUT_DIR, (os.path.splitext(item)[0]+'.pt')))

            if SHOW:
                img = img.permute(1, 2, 0).cpu().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
