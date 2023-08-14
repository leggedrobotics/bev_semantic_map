#!/usr/bin/env python

"""
Stores the file names to a txt file. Fills time stamps with zeros in the end.
Use to extract points at the desired time stamps.

Author: Robin Schmid
Date: Nov 2022
"""

import os
from tqdm import tqdm

# Path to image directory
DATASET = "hoengg_vision_train"
INPUT_DIR = f"/home/rschmid/RosBags/{DATASET}/image"

if __name__ == "__main__":

    with open(f"../../../docs/timestamps/{DATASET}.txt", "w") as output:
        for path, subdirs, files in os.walk(INPUT_DIR):
            files.sort()
            for filename in tqdm(files):
                filename = str(filename).replace(".pt", "")
                filename = filename.replace("_", "")

                # Want filenames to have a length of 17, fill from the end with zeros
                filename = filename[::-1].zfill(17)[::-1]
                filename = filename[:-1]
                output.write(filename + os.linesep)
