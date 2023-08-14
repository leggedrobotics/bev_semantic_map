#!/usr/bin/env python

"""
Shows torch images.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
import sys

# data_path = "/home/rschmid/RosBags/helendale_train/supervision_mask"
data_path = "/home/rschmid/RosBags/hoengg_vision_train/supervision_mask"

np.set_printoptions(threshold=sys.maxsize)

file_names = [os.path.basename(d) for d in sorted(glob.glob(data_path + "/*"))]

for file_name in file_names:

    img = torch.load(os.path.join(data_path, file_name), map_location=torch.device('cpu'))


    print(img.cpu().numpy())

    print(img.shape)
    print(img.dtype)

    plt.title(file_name)
    plt.imshow(img)
    plt.show()
