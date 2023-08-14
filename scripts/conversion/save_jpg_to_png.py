#!/usr/bin/env python

"""
Saves .jpg to .png files or vice versa.

Author: Robin Schmid
Date: Nov 2022
"""


import os
from PIL import Image

"""
Converts all ".jpg" images in a directory to '.png' format or vice versa.
"""

# Path to image directory, do not forget the "/" in the end!
INPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask_png"
OUTPUT_DIR = "/home/rschmid/RosBags/arroyo_train/supervision_mask_jpg"
OUTPUT_EXT = 'jpg'  # Select png or jpg
items = os.listdir(INPUT_DIR)
items.sort()

print("Num items found", len(items))


if __name__ == "__main__":

    for i, item in enumerate(items):
        path = os.path.join(INPUT_DIR, item)
        if os.path.isfile(path):
            print(i)
            print(item)
            try:
                img = Image.open(path)
                img.save(os.path.join(OUTPUT_DIR, (os.path.splitext(item)[0] + '.' + OUTPUT_EXT)))
            except:
                print("pass")
                pass
