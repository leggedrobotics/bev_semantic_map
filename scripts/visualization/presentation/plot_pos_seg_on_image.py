#!/usr/bin/env python

"""
Plot mask on the image as an overlay wtih segmentation lines.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import torch
import cv2
import glob
import numpy as np
from tqdm import tqdm
import kornia as K

IMG_PATH = "/home/rschmid/Documents/final_presentation/approach/2/points"
SEG_PATH = "/home/rschmid/Documents/final_presentation/approach/2/seg"
MASK_PATH = "/home/rschmid/Documents/final_presentation/approach/2/supervision_mask"

# IMG_PATH = "/home/rschmid/RosBags/perugia_100_eval/image"
# MASK_PATH = "/home/rschmid/RosBags/perugia_100_eval/labels"

VISUALIZE = True

if __name__ == "__main__":
    file_names = [os.path.basename(d) for d in sorted(glob.glob(IMG_PATH + "/*"))]

    for file_name in tqdm(file_names):
        # img = torch.load(os.path.join(IMG_PATH, file_name), map_location=torch.device('cpu'))\
        #     .permute(1, 2, 0).cpu().numpy()
        img = torch.from_numpy(cv2.imread(os.path.join(IMG_PATH, file_name))).cpu().numpy()
        img = img.astype(np.float32)
        img /= 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.imread(os.path.join(IMG_PATH, file_name))
        seg = torch.load(SEG_PATH + "/" +
                         os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu')).cpu()
        mask = torch.load(MASK_PATH + "/" +
                          os.path.splitext(file_name)[0] + ".pt", map_location=torch.device('cpu')).cpu().numpy()

        # Iterate through all segments of the segmentation
        overlay = img.copy()

        for s in torch.unique(seg):
            m = mask[seg == s]
            pos = m.sum()  # Inside
            neg = (~m).sum()  # Outside

            # Check if more inside features than outside features
            if pos > neg:
                overlay[seg == s] = [0, 1.0, 0]

        seg = seg.unsqueeze(0).unsqueeze(0).type(torch.float32)

        seg_lines: torch.Tensor = K.filters.canny(seg)[0]
        seg_lines = seg_lines.clamp(0., 1.)
        seg_lines = seg_lines.squeeze(0).squeeze(0).numpy()

        # Add additional dimensions to make one channel image to three channel image
        seg_lines = np.expand_dims(seg_lines, axis=2)
        seg_lines = np.concatenate((seg_lines, seg_lines, seg_lines), axis=2)

        inter = cv2.addWeighted(overlay, 0.6, img, 0.4, 0.0)

        res = cv2.addWeighted(seg_lines, 0.2, inter, 0.8, 0.0)

        # For torch images
        res = cv2.convertScaleAbs(res, alpha=(255.0))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        if VISUALIZE:
            cv2.imshow("overlay", res)
            cv2.waitKey(0)

        cv2.imwrite(f"/home/rschmid/{os.path.splitext(os.path.basename(file_name))[0]}.jpg", res)
