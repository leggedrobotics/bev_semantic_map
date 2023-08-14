#!/usr/bin/env python

"""
Prints an .pt image in the terminal

Author: Robin Schmid
Date: Feb 2023
"""

import torch
import sys
import numpy as np

torch.set_printoptions(edgeitems=100)

IMG_PATH = "/home/rschmid/RosBags/helendale_eval/supervision_mask/1635970051266645000.pt"
IMG_PATH2 = "/home/rschmid/RosBags/hoengg_vision_train/supervision_mask/1638801188_2023723.pt"

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    img = torch.load(IMG_PATH2, map_location=torch.device('cpu'))

    print(img.dtype)

    print(img.shape)

    # print(img)
