#!/usr/bin/env python

"""
BEV anomaly detection for safe and unsafe traversability prediction.

Author: Robin Schmid
Date: Sep 2023
"""

import os
import cv2
import numpy as np
import torch
import wandb
import argparse
from tqdm import tqdm

from bevnet.cfg import ModelParams, RunParams, DataParams
from bevnet.network.bev_net import BevNet
from bevnet.dataset import get_bev_dataloader
from bevnet.utils import Timer, DataVisualizer

# Global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=200)

POS_WEIGHT = 0.2  # Num neg / num pos
THRESHOLD = 0.1
VISU_DATA = False


class BevTraversability:
    def __init__(self, wandb_logging=False):
        self._model_cfg = ModelParams()
        self._run_cfg = RunParams()
        self._data_cfg = DataParams()

        self._model = BevNet(self._model_cfg)
        self._model.cuda()

        self.wandb_logging = wandb_logging
        if self.wandb_logging:
            wandb.init(project=self._run_cfg.log_name)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._run_cfg.lr)

        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))

        if VISU_DATA:
            self.data_visu = DataVisualizer()

    def train(self, save_model=False, model_name="bevnet"):
        self._model.train()

        data_loader, _ = get_bev_dataloader(mode="train", batch_size=self._run_cfg.training_batch_size)

        for _ in tqdm(range(self._run_cfg.epochs)):
            for j, batch in enumerate(data_loader):
                imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
                pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                    pcd_new["points"].cuda(),
                    pcd_new["batch"].cuda(),
                    pcd_new["scan"].cuda(),
                )

                if VISU_DATA:
                    # ROS visualization
                    # target_out = target.numpy() + 1
                    # target_out = target_out.astype(np.uint8).squeeze(1)
                    #
                    # pc_out = self.data_visu.correct_z_direction(pcd_new["points"])
                    #
                    # self.data_visu.publish_occ_map(target_out, res=0.1)
                    # self.data_visu.publish_pc(pc_out)

                    # OpenCV visualization
                    target_out = target.squeeze()
                    target_out = target_out + 1
                    vis_img = np.zeros((target_out.shape[0], target_out.shape[1], 3), dtype=np.uint8)

                    vis_img[target_out == 0] = (0, 0, 0)  # Unknown
                    vis_img[target_out == 1] = (0, 255, 0)  # Safe
                    vis_img[target_out == 2] = (0, 0, 255)  # Unsafe

                    cv2.imshow("img", vis_img)
                    cv2.waitKey(0)

                # Forward pass
                pred = self._model(
                    imgs.cuda(),
                    rots.cuda(),
                    trans.cuda(),
                    intrins.cuda(),
                    post_rots.cuda(),
                    post_trans.cuda(),
                    target.cuda().shape,
                    pcd_new,
                    target.cuda(),
                )

                # Compute loss
                mask = (target > -1).float().cuda()
                loss = self._loss(pred, target.cuda())
                loss = loss * mask
                num_pixels = mask.sum()
                if num_pixels > 0:
                    loss_mean = loss.sum() / num_pixels  # Average loss over all non-background pixels
                else:
                    loss_mean = torch.tensor(0.0)  # Avoid division by zero if there are no non-background pixels

                print(f"{j} | {loss_mean.item():.9f}")

                if self.wandb_logging:
                    wandb.log({"train_loss": loss_mean.item()})

                # Backward pass
                self._optimizer.zero_grad()
                loss_mean.backward()
                self._optimizer.step()

            if save_model:
                print("Saving model ...")
                torch.save(self._model.state_dict(), f"bevnet/weights/{model_name}.pth")

    def predict(self, load_model=True, model_name="bevnet", save_pred=False):
        if load_model:
            self._model.to(DEVICE)
            try:
                self._model.load_state_dict(
                    torch.load(f"bevnet/weights/{model_name}.pth", map_location=torch.device(DEVICE)), strict=True
                )
            except:
                ValueError("This model configuration does not exist!")

            # Set the model to evaluation mode
            self._model.eval()

        data_loader, self._data_cfg.data_dir = get_bev_dataloader(mode=TEST_DATASET, batch_size=1)
        for j, batch in enumerate(data_loader):
            imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
            pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                pcd_new["points"].cuda(),
                pcd_new["batch"].cuda(),
                pcd_new["scan"].cuda(),
            )
            with Timer(f"Inference {j}"):
                with torch.no_grad():
                    # Forward pass
                    pred = self._model(
                        imgs.cuda(),
                        rots.cuda(),
                        trans.cuda(),
                        intrins.cuda(),
                        post_rots.cuda(),
                        post_trans.cuda(),
                        target.cuda().shape,
                        pcd_new,
                    )

            # Apply sigmoid and convert to NumPy array
            x = torch.sigmoid(pred).squeeze().cpu().detach().numpy()

            if save_pred:
                # x = np.flip(x, axis=0) # Flip to match the image
                torch.save(x, os.path.join(self._data_cfg.data_dir, "pred", f"{j:04d}.pt"))


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true", help="""If set trains""")
    parser.add_argument("-p", action="store_true", help="""If set predicts""")
    parser.add_argument("-l", action="store_true", help="""Logs data on wandb""")
    parser.add_argument('-d', default='p', choices=['t', 'p'], help="""Dataset specified (t: train, p: pred / test)""")
    args = parser.parse_args()

    bt = BevTraversability(args.l)

    if args.d == 't':
        TEST_DATASET = "train"
    else:
        TEST_DATASET = "test"

    # Setting training mode
    if args.t:
        bt.train(save_model=True)
    elif args.p:
        bt.predict(load_model=True, save_pred=True)
    else:
        raise ValueError(f"Unknown mode, please specify -t (for train mode), -p (for test mode)")
