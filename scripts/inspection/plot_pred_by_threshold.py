#!/usr/bin/env python

"""
Plots predictions with variable threshold and interactivity.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import glob
import cv2
import numpy as np
import torch

# Both img and pred are torch tensors
IMG_PATH = "/home/rschmid/RosBags/hoengg_geom_train_vel/image"
PRED_PATH = "/home/rschmid/git/wild_anomaly_detection/samples/pred/train=hoengg_geom_train_vel_2000_test=hoengg_geom_train_vel_2000_bs=200_eps=5_nf=10_top=100_th=0.4"


class PredPLotter():
    def __init__(self):
        self.counter = 0

    def nothing(self, x):
        pass

    def back(self, *args):
        self.counter -= 1

    def forward(self, *args):
        self.counter += 1

    def run(self):

        # Plotting params
        THRESHOLD = 30  # Threshold for safe region; lower: more is safe; higher: less is safe. Between 0 and 100.
        OVERLAY_ALPHA = 0.35  # Overlay of traversability mask. Between 0 and 1.
        RESIZE = 0.7  # Resize factor for image. Between 0 and 1 such that it can fit on the screen.

        img_paths = sorted(glob.glob(os.path.join(IMG_PATH, "*")))
        pred_paths = sorted(glob.glob(os.path.join(PRED_PATH, "*")))

        print("Number of images:", len(img_paths))

        # Start loop with the images
        while True:
            print(self.counter)

            if self.counter >= len(img_paths) or self.counter < 0:
                exit()

            # Create window and interactive buttons
            cv2.namedWindow('image')

            cv2.createButton("Back", self.back, None, cv2.QT_PUSH_BUTTON, 1)
            cv2.createButton("Forward", self.forward, None, cv2.QT_PUSH_BUTTON, 1)

            cv2.createTrackbar('threshold', 'image', 0, 100, self.nothing)
            cv2.setTrackbarPos('threshold', 'image', THRESHOLD)

            # Start the loop
            while True:
                imgs = []

                # Fix threshold if out of bounds
                THRESHOLD = THRESHOLD / 100.0
                if THRESHOLD >= 1.0:
                    THRESHOLD = 0.99
                elif THRESHOLD <= 0.0:
                    THRESHOLD = 0.01

                for j in range(self.counter, self.counter + 9):
                    img = torch.load(img_paths[j], map_location="cpu").permute(1, 2, 0).cpu().numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    pred = torch.load(pred_paths[j], map_location="cpu")

                    # Safe region
                    overlay = img.copy()
                    overlay[:] = [0.0, 1.0, 0.0] - np.multiply.outer(
                        pred * (1 / THRESHOLD),
                        [0.0, 1.0, 0.0],
                    )
                    overlay[pred >= THRESHOLD] = 0
                    # Unsafe regions
                    pred[pred < THRESHOLD] = 0
                    overlay[:] += np.multiply.outer(
                        pred * (1 / (1 - THRESHOLD)),
                        [0.0, 0.0, 1.0],
                    )

                    # Only plot generate overlay for non-NaN values
                    overlay[np.isnan(pred)] = img[np.isnan(pred)]

                    img = cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1 - OVERLAY_ALPHA, 0.0)

                    # Resize image
                    img = cv2.resize(img, (448, 448), fx=RESIZE, fy=RESIZE)

                    imgs.append(img)

                # combine images
                combined = np.vstack((np.hstack((imgs[0], imgs[1], imgs[2])),
                                      np.hstack((imgs[3], imgs[4], imgs[5])),
                                      np.hstack((imgs[6], imgs[7], imgs[8]))))

                cv2.imshow("9 consecutive predictions", combined)
                cv2.waitKey(1)

                THRESHOLD = cv2.getTrackbarPos('threshold', 'image')

                print("Threshold:", THRESHOLD)


if __name__ == "__main__":
    p = PredPLotter()
    p.run()
