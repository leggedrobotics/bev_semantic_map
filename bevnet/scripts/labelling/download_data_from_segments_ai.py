#!/usr/bin/env python

"""
Downloads the data from segments.ai and saves it.

Author: Robin Schmid
Date: Aug 2023
"""

from segments import SegmentsClient
from segments.utils import load_label_bitmap_from_url
import torch
import cv2
import numpy as np
from tqdm import tqdm

key = "19fb088f9cb12bd279c129c800df6d10b161ae3e"
client = SegmentsClient(key)

data_set_name = "occupancy_map"

dataset_identifier = "schmirob" + "/" + data_set_name
samples = client.get_samples(dataset_identifier)
for sample in tqdm(samples):
    label = client.get_label(sample.uuid)
    res = load_label_bitmap_from_url(label.attributes.segmentation_bitmap.url)
    # n = sample.name.replace(".jpg", ".png")
    # n = sample.name.replace(".png", ".pt")
    # res = torch.from_numpy((res - 1).astype(bool))
    # torch.save(res, f"/home/rschmid/{n}")

    gray = cv2.cvtColor(np.float32(res), cv2.COLOR_GRAY2RGB)
    n = sample.name

    # n = sample.name
    cv2.imwrite(f"/home/rschmid/{n}", gray)
