#!/usr/bin/env python

"""
Plot mask on the image as an overlay.

Author: Robin Schmid
Date: Dec 2022
"""

import os
import sys
import glob
from tqdm import tqdm
import cv2
import numpy as np
import torch
import kornia as K

IMG_PATH = "/home/rschmid/img"
SEG_PATH = "/home/rschmid/2000_seg"

np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]

    for file_name in tqdm(file_names):
        img = torch.load(os.path.join(IMG_PATH, file_name), map_location=torch.device('cpu')).permute(1, 2, 0).cpu().numpy()
        # img = torch.from_numpy(cv2.imread(os.path.join(IMG_PATH, file_name))).cpu().numpy()
        # img = img.astype(np.float32)
        # img /= 255.0
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = torch.load(SEG_PATH + "/" + os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu')).cpu()
        seg = seg.unsqueeze(0).unsqueeze(0).type(torch.float32)

        # Alternative approaches to compute the edges
        # overlay: torch.Tensor = K.filters.sobel(seg)

        # overlay: torch.Tensor = K.filters.laplacian(seg, kernel_size=5)
        # overlay = overlay.clamp(0, 1)

        overlay: torch.Tensor = K.filters.canny(seg)[0]
        overlay = overlay.clamp(0., 1.)

        overlay = overlay.squeeze(0).squeeze(0).numpy()

        # Add additional dimensions to make one channel image to three channel image
        overlay = np.expand_dims(overlay, axis=2)
        overlay = np.concatenate((overlay, overlay, overlay), axis=2)

        res = cv2.addWeighted(overlay, 0.3, img, 0.7, 0.0)

        # For torch images
        res = cv2.convertScaleAbs(res, alpha=(255.0))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        cv2.imshow("overlay", res)
        cv2.waitKey(0)

        cv2.imwrite(f"/home/rschmid/{os.path.splitext(os.path.basename(file_name))[0]}.png", res)
