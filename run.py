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
import time
import wandb
import argparse
from tqdm import tqdm
from icecream import ic

from bevnet.cfg import ModelParams, RunParams, DataParams
from bevnet.network.bev_net import BevNet
from bevnet.models import AutoEncoder
from bevnet.loss import AnomalyLoss
from bevnet.dataset import get_bev_dataloader
from bevnet.utils import Timer

# Global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=200)


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

        if self._model_cfg.fusion_backbone == "CNN":
            self._loss = torch.nn.functional.mse_loss
        elif self._model_cfg.fusion_backbone == "RNVP":
            self._loss = AnomalyLoss()
        elif self._model_cfg.fusion_backbone == "MLP":
            self._loss = torch.nn.functional.mse_loss
        if self._model_cfg.autoencoder:
            self._autoencoder = AutoEncoder()
            self._autoencoder.cuda()

    def train(self, save_model=False, model_name="bevnet"):
        self._model.train()

        data_loader = get_bev_dataloader(mode="train", batch_size=self._run_cfg.training_batch_size)

        for _ in tqdm(range(self._run_cfg.epochs)):
            for j, batch in enumerate(data_loader):
                imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
                pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                    pcd_new["points"].cuda(),
                    pcd_new["batch"].cuda(),
                    pcd_new["scan"].cuda(),
                )

                # Forward pass
                pred, pred_ae = self._model(
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
                if self._model_cfg.fusion_backbone == "CNN":
                    mask = (target.cuda() > 0).float()
                    loss_mean = self._loss(pred, target.cuda().float(), reduction="none") * mask
                    loss_mean = torch.mean(loss_mean[loss_mean != 0])
                    # loss_mean = self._loss(pred[target.cuda()], pred_ae[target.cuda()])
                elif self._model_cfg.fusion_backbone == "RNVP":
                    loss_mean, loss_pred = self._loss(pred)
                elif self._model_cfg.fusion_backbone == "MLP":
                    loss_mean = self._loss(pred, target.cuda().float().reshape(-1, 1))

                print(f"{j} | {loss_mean.item():.5f}")

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

        data_loader = get_bev_dataloader(mode="test", batch_size=1)
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
                    pred, pred_ae = self._model(
                        imgs.cuda(),
                        rots.cuda(),
                        trans.cuda(),
                        intrins.cuda(),
                        post_rots.cuda(),
                        post_trans.cuda(),
                        target.cuda().shape,
                        pcd_new,
                    )

            # Compute loss
            if self._model_cfg.fusion_backbone == "RNVP":
                _, x = self._loss(pred)
            else:
                x = pred

            # Normalize for visualization
            x = x.cpu().detach().numpy()
            sz = int(x.size ** 0.5)
            x = x.reshape(sz, sz)   # From (B=1, C=1, H, W) to (H, W)
            pred_out = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

            if save_pred:
                cv2.imwrite(os.path.join(os.path.split(self._data_cfg.data_dir)[0],
                                         "pred", f"{j:04d}.jpg"), pred_out)
                if self.wandb_logging:
                    wandb.log({"prediction": wandb.Image(pred_out)})


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true", help="""If set trains""")
    parser.add_argument("-p", action="store_true", help="""If set predicts""")
    parser.add_argument("-l", action="store_true", help="""Logs data on wandb""")
    args = parser.parse_args()

    bt = BevTraversability(args.l)

    # Setting training mode
    if args.t:
        bt.train(save_model=True)
    elif args.p:
        bt.predict(load_model=True, save_pred=True)
    else:
        raise ValueError(f"Unknown mode, please specify -t (for train), -p (for test), -l (for wandb logging)")
