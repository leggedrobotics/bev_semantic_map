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
    def __init__(self):
        self._model_cfg = ModelParams()
        self._model = BevNet(self._model_cfg)
        self._model.cuda()

        self._run_cfg = RunParams()
        if self._run_cfg.wandb_logging:
            wandb.init(project="bevnet")

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._model_cfg.fusion_net.lr)
        self._loss = AnomalyLoss()

    def train(self, save_model=False):
        self._model.train()

        loader_train, _ = get_bev_dataloader(batch_size=self._run_cfg.training_batch_size)
        for j, batch in enumerate(loader_train):
            imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
            pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                pcd_new["points"].cuda(),
                pcd_new["batch"].cuda(),
                pcd_new["scan"].cuda(),
            )
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

            loss_mean, loss_pred = self._loss(pred)

            print(f"{j} | {loss_mean.item():.5f}")

            if self._run_cfg.wandb_logging:
                wandb.log({"train_loss": loss_mean.item()})

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

        _, _, loader_test = get_bev_dataloader(return_test_dataloader=True, batch_size=1)
        for j, batch in enumerate(loader_test):
            imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
            pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                pcd_new["points"].cuda(),
                pcd_new["batch"].cuda(),
                pcd_new["scan"].cuda(),
            )
            with Timer(f"Inference {j}"):
                with torch.no_grad():
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

            loss_mean, loss_pred = self._loss(pred)

            # print(loss_train)
            x = loss_pred.cpu().detach().numpy()
            square_size = int(x.size ** 0.5)
            x = x.reshape(square_size, square_size)

            pred = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

            if save_pred:
                cv2.imwrite(f"/home/rschmid/RosBags/bevnet/pred/{j}.jpg", pred)

                if self._run_cfg.wandb_logging:
                    wandb.log({"prediction": wandb.Image(pred)})


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true',
                        help="""If set trains""")
    parser.add_argument("--pred", action='store_true',
                        help="""If set predicts""")
    parser.add_argument("--eval", action='store_true',
                        help="""If set evaluates""")
    args = parser.parse_args()

    bt = BevTraversability()

    # Setting training mode
    if args.train:
        bt.train(save_model=True)
    elif args.pred:
        bt.predict(load_model=True, save_pred=True)
    else:
        raise ValueError(f"Unknown mode")
