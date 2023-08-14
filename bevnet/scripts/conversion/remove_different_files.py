#!/usr/bin/env python

"""
Removes files from a directory which are not contained in another directory.

Author: Robin Schmid
Date: Jan 2023
"""


import os

REVERSE = True

# Set the paths for the two directories
dir1 = "/home/rschmid/RosBags/output/perugia_grass/supervision_mask"  # The directory which should be cleaned
dir2 = "/home/rschmid/RosBags/output/perugia_grass/image"

# Get the list of files in both directories
files1 = set(os.listdir(dir1))
ending_files1 = os.path.splitext(list(files1)[0])[1]
# Remove endings in files1
files1 = {os.path.splitext(file)[0] for file in files1}

files2 = set(os.listdir(dir2))
ending_files2 = os.path.splitext(list(files2)[0])[1]
# Remove endings in files2
files2 = {os.path.splitext(file)[0] for file in files2}

# Find the files that are in dir1 but not in dir2
files_to_delete = files1 - files2

# Iterate through the files to delete and remove them in dir1
for file_name in files_to_delete:
    file_path = os.path.join(dir1, file_name) + ending_files1
    os.remove(file_path)

print(f"Removed {len(files_to_delete)} files")

if REVERSE:
    # Now do the reverse, find the files that are in dir2 but not in dir1
    files_to_delete = files2 - files1

    # Iterate through the files to delete and remove them in dir1
    for file_name in files_to_delete:
        file_path = os.path.join(dir2, file_name) + ending_files2
        os.remove(file_path)

    print(f"Removed {len(files_to_delete)} files")

print("Done")
