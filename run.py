#!/usr/bin/env python

"""
BEV anomaly detection for safe and unsafe traversability prediction.

Author: Robin Schmid
Date: Sep 2023
"""


import cv2
import numpy as np
import torch
import wandb
import argparse
from tqdm import tqdm
from icecream import ic

from bevnet.cfg import ModelParams, RunParams
from bevnet.network.bev_net import BevNet
from bevnet.network.loss import AnomalyLoss
from bevnet.dataset import get_bev_dataloader
from bevnet.utils import Timer

# Global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=200)


class BevTraversability:
    def __init__(self, wandb_logging=False):
        self._model_cfg = ModelParams()
        self._model = BevNet(self._model_cfg)
        self._model.cuda()

        self.wandb_logging = wandb_logging

        self._run_cfg = RunParams()
        if self.wandb_logging:
            wandb.init(project="bevnet")

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._model_cfg.fusion_net.lr)
        # self._loss = AnomalyLoss()
        self._loss = torch.nn.functional.mse_loss

    def train(self, save_model=False):
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
                pred = self._model(
                    imgs.cuda(),
                    rots.cuda(),
                    trans.cuda(),
                    intrins.cuda(),
                    post_rots.cuda(),
                    post_trans.cuda(),
                    target.cuda().shape,
                    pcd_new,
                    target.cuda()
                )

                # Compute loss
                # loss_mean, loss_pred = self._loss(pred, target.cuda())
                loss_mean = self._loss(pred, target.cuda().float())

                print(f"{j} | {loss_mean.item():.5f}")

                if self.wandb_logging:
                    wandb.log({"train_loss": loss_mean.item()})

                # Backward pass
                self._optimizer.zero_grad()
                loss_mean.backward()
                self._optimizer.step()

            if save_model:
                print("Saving model ...")
                torch.save(self._model.state_dict(), "bevnet/weights/bevnet.pth")

    def predict(self, load_model=True, model_name="bevnet", save_pred=False):
        if load_model:
            self._model.to(DEVICE)
            try:
                self._model.load_state_dict(
                    torch.load(f"bevnet/weights/{model_name}.pth", map_location=torch.device(DEVICE)), strict=False)
            except:
                ValueError("This model configuration does not exist!")

            # Set the model to evaluation mode
            # self._model.eval()    # TODO: turning this on causes different output with big values

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

            # loss_mean, loss_pred = self._loss(pred)

            # print(loss_train)
            x = pred.cpu().detach().numpy()
            square_size = int(x.size ** 0.5)
            x = x.reshape(square_size, square_size)
            pred = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

            # ic(pred)

            if save_pred:
                cv2.imwrite(f"/home/rschmid/RosBags/bevnet/pred/{j}.jpg", pred)

                if self.wandb_logging:
                    wandb.log({"prediction": wandb.Image(pred)})


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", action='store_true',
                        help="""If set trains""")
    parser.add_argument("--p", action='store_true',
                        help="""If set predicts""")
    parser.add_argument("--log", action='store_true',
                        help="""Logs data on wandb""")
    args = parser.parse_args()

    bt = BevTraversability(args.log)

    # Setting training mode
    if args.t:
        bt.train(save_model=True)
    elif args.p:
        bt.predict(load_model=True, save_pred=True)
    else:
        raise ValueError(f"Unknown mode")
