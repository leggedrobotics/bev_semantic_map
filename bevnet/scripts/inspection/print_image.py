#!/usr/bin/env python

"""
Prints an .jpg image in the terminal

Author: Robin Schmid
Date: Jan 2023
"""

import sys
import numpy as np
import cv2

IMG_PATH = "/home/rschmid/Downloads/000003.png"

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    img = cv2.imread(IMG_PATH)

    print(img)

    # print(img)
