#!/usr/bin/env python

"""
BEV anomaly detection for safe and unsafe traversability prediction.

Author: Robin Schmid
Date: Sep 2023
"""


import cv2
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


class BevTraversability:
    def __init__(self):
        self._model_cfg = ModelParams()
        self._model = BevNet(self._model_cfg)
        self._model.cuda()

        self._run_cfg = RunParams()
        if self._run_cfg.wandb_logging:
            wandb.init(project="bevnet")

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._model_cfg.fusion_net.lr)
        # self.optimizer = torch.optim.Adam(self.fusion_net.parameters(), lr=cfg_model.fusion_net.lr)
        self._loss = AnomalyLoss()

    def train(self, save_model=False):
        self._model.train()

        loader_train, _ = get_bev_dataloader(batch_size=1)
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

            loss, loss_pred = self._loss(pred)
            # print(loss_pred)

            print(f"{j} | {loss.item():.5f}")

            if self._run_cfg.wandb_logging:
                wandb.log({"train_loss": loss.item()})

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

        if save_model:
            torch.save(self._model.state_dict(), "bevnet/weights/bevnet.pth")

    def predict(self, load_model=True, model_name="bevnet", save_pred=False):
        if load_model:
            self._model.to(DEVICE)
            try:
                self._model.load_state_dict(
                    torch.load(f"bevnet/weights/{model_name}.pth", map_location=torch.device(DEVICE)))
            except:
                ValueError("This model configuration does not exist!")

            # Set the model to evaluation mode
            self._model.eval()

        _, _, loader_test = get_bev_dataloader(return_test_dataloader=True)
        for j, batch in enumerate(loader_test):
            # print(j)
            imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
            pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                pcd_new["points"].cuda(),
                pcd_new["batch"].cuda(),
                pcd_new["scan"].cuda(),
            )
            with Timer(f"Inference {j}"):
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

            _, loss = self._loss(pred)

            print(loss)
            # pred = losses.view(128, 128)
            # print(pred)

            # if save_pred:
            #     # Save predictions as grayscale images
            #     pred = pred.cpu().detach().numpy()
            #     pred_out = pred[0, 0] * 255
            #
            #     cv2.imwrite(f"data/pred/{j}.jpg", pred_out)
            #
            #     if self._run_cfg.wandb_logging:
            #         wandb.log({"prediction": wandb.Image(pred_out)})


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

    if args.train:
        bt.train(save_model=True)
    elif args.pred:
        bt.predict(load_model=True, save_pred=True)
    else:
        raise ValueError(f"Unknown mode")
