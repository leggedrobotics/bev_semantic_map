#!/usr/bin/env python

"""
Creates a gif from a folder with images.

Author: Robin Schmid
Date: Oct 2022
"""

import glob
from PIL import Image

SCALE = 0.5  # Scale the images to reduce the file size of the gif. Between 0 and 1.
INPUT_FOLDER = "/home/rschmid/gif"
OUTPUT_FILE_NAME = "/home/rschmid/gif"

DURATION = 200  # Duration of each frame in the gif in ms
NUM_FILES = 100  # Number of files to use for the gif. None for all files.


def make_gif(frame_folder, file_name, scale=None, num_files=NUM_FILES):
    files = sorted(glob.glob(f"{frame_folder}/*"))
    if num_files is not None:
        files = files[:num_files]

    input_size = Image.open(files[0]).size
    frames = [Image.open(image).resize((int(input_size[0] * scale), int(input_size[1] * scale)))
              for image in files]
    # frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*." + extension))]
    frame_one = frames[0]
    frame_one.save(f"{file_name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=DURATION, loop=0)


if __name__ == "__main__":
    make_gif(frame_folder=INPUT_FOLDER,
             file_name=OUTPUT_FILE_NAME, scale=SCALE)
    print("Done")
