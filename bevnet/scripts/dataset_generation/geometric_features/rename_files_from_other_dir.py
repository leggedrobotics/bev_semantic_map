#!/usr/bin/env python

"""
Renames files from a source directory to a target directory. Iterates the files sequentially.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import glob

PATHS_TO_CHANGE = "/home/rschmid/RosBags/hoengg_geom_train_vel/torch_features"
DESIRED_PATH = "/home/rschmid/RosBags/hoengg_geom_train_vel/supervision_mask"

if __name__ == "__main__":

    old_paths = sorted(glob.glob(os.path.join(PATHS_TO_CHANGE, "*")))
    new_paths = sorted(glob.glob(os.path.join(DESIRED_PATH, "*")))

    for i in range(len(new_paths)):
        # Rename the old paths with the new paths
        os.rename(old_paths[i], os.path.join(os.path.split(old_paths[i])[0], os.path.split(new_paths[i])[1]))

        # Print the new paths
        print(os.path.join(os.path.split(old_paths[i])[0], os.path.split(new_paths[i])[1]))
