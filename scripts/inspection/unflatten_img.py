#!/usr/bin/env python

"""
Flattens and unflattens a tensor for visualization.

Author: Robin Schmid
Date: Feb 2023
"""

import torch
import cv2

DATASET = "perugia_seg_pix"

IMG_PATH = "/home/rschmid/RosBags/perugia_seg_pix_single/image/1652349958_431409.pt"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    print("Saving positive features")

    img = torch.load(IMG_PATH, map_location=DEVICE)

    print(img.shape)

    # cv2.imshow("img", cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    img = img.flatten(start_dim=1, end_dim=2)

    print(img.shape)

    img = img.view(3, 224, 224)

    print(img.shape)

    # cv2.imshow("img", cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
