#!/usr/bin/env python

"""
BEVnet for safe and unsafe traversability prediction.

Author: Robin Schmid
Date: Sep 2023
"""

import os
import cv2
import time
import json
import numpy as np
import torch
import wandb
import argparse
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bevnet.cfg import ModelParams, RunParams, DataParams
from bevnet.network.bev_net import BevNet
from bevnet.dataset import get_bev_dataloader
from bevnet.utils import Timer

# Global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=200)

MODEL_NAME = None
# MODEL_NAME = "2023_12_13_10_43_22"

POS_WEIGHT = 10  # Num neg / num pos, from data around 0.08
THRESHOLD = 0.1
VISU_DATA = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "bevnet", "data")


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

        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._run_cfg.lr)

        self._loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]), ignore_index=-1)
        self._loss.cuda()

    def train(self, save_model=False):
        self._model.train()

        model_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        # Create data path
        if not os.path.exists(os.path.join(DATA_PATH, model_name)):
            os.makedirs(os.path.join(DATA_PATH, model_name, "config"))
            os.makedirs(os.path.join(DATA_PATH, model_name, "weights"))
            os.makedirs(os.path.join(DATA_PATH, model_name, "pred_train"))
            os.makedirs(os.path.join(DATA_PATH, model_name, "pred_test"))
            os.makedirs(os.path.join(DATA_PATH, model_name, "pred_train_epochs"))

        # Save configs as json
        with open(os.path.join(DATA_PATH, model_name, "config", "model_params.json"), 'w') as json_file:
            json.dump(asdict(self._model_cfg), json_file, indent=2)
        with open(os.path.join(DATA_PATH, model_name, "config", "run_params.json"), 'w') as json_file:
            json.dump(asdict(self._run_cfg), json_file, indent=2)
        with open(os.path.join(DATA_PATH, model_name, "config", "data_params.json"), 'w') as json_file:
            json.dump(asdict(self._data_cfg), json_file, indent=2)

        data_loader, _ = get_bev_dataloader(mode="train", batch_size=self._run_cfg.training_batch_size, shuffle=True)

        num_data = len(data_loader)

        for i in range(self._run_cfg.epochs):
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
                    target.cuda(),
                )

                pred = pred.softmax(dim=1).float()
                target = target.float()

                # Compute loss
                loss_mean = self._loss(pred, target.long().cuda()[:, 0, :, :])

                print(f"Epoch {i} / {self._run_cfg.epochs} | Batch {j} / {num_data} | Loss {loss_mean.item():.9f}")

                if self.wandb_logging:
                    wandb.log({"train_loss": loss_mean.item()})

                # Backward pass
                self._optimizer.zero_grad()
                loss_mean.backward()
                self._optimizer.step()

            if VISU_DATA:
                # Convert tensors to numpy arrays
                pred_np = pred.clone().squeeze().cpu().detach().numpy()
                target_np = target.clone().squeeze().cpu().detach().numpy()
                
                # Check case if there is not batch dimension
                if len(pred_np.shape) == 2:
                    pred_np = pred_np[np.newaxis, ...]
                if len(target_np.shape) == 2:
                    target_np = target_np[np.newaxis, ...]

                # Normalize between v_min and v_max
                v_min, v_max = 0, 1

                # Replace -1 with np.nan
                pred_np[pred_np == -1] = np.nan
                target_np[target_np == -1] = np.nan

                cmap = sns.color_palette("RdYlBu", as_cmap=True)

                cmap.set_bad(color="black")

                # Plot images side by side
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                b = 0
                plt.imshow(pred_np[b, 1], cmap='coolwarm', vmin=v_min, vmax=v_max)
                plt.title('Pred')
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.imshow(target_np[b], cmap='coolwarm', vmin=v_min, vmax=v_max)
                plt.title('Target')
                plt.colorbar()

                # Save to disk
                plt.savefig(os.path.join(DATA_PATH, model_name, "pred_train_epochs",
                                         f'{i}.png'))
                plt.close()

            if save_model:
                print(f"Saving model ... {model_name}")
                torch.save(self._model.state_dict(),
                           os.path.join(DATA_PATH, model_name, "weights", f"{model_name}.pth"))

    def predict(self, load_model=True, model_name=None, save_pred=False, test_dataset=None):
        if load_model:
            self._model.to(DEVICE)

            # If no specific name is given load the latest model
            if model_name is None:
                all_directories = [d for d in os.listdir(DATA_PATH) if
                                   os.path.isdir(os.path.join(DATA_PATH, d))]
                sorted_directories = sorted(all_directories)
                model_name = sorted_directories[-1]

            print(f"Using model from {model_name}")

            try:
                self._model.load_state_dict(
                    torch.load(
                        os.path.join(DATA_PATH, model_name, "weights", f"{model_name}.pth"),
                        map_location=torch.device(DEVICE)), strict=True
                )
            except:
                ValueError("This model configuration does not exist!")

            if not os.path.exists(os.path.join(DATA_PATH, model_name, f"pred_{test_dataset}")):
                os.makedirs(os.path.join(DATA_PATH, model_name, f"pred_{test_dataset}"))

            # Set the model to evaluation mode
            self._model.eval()

        data_loader, self._data_cfg.data_dir = get_bev_dataloader(mode=test_dataset, batch_size=1)
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
                torch.save(x, os.path.join(DATA_PATH, model_name, f"pred_{test_dataset}", f"{j:04d}.pt"))


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true", help="""If set trains""")
    parser.add_argument("-p", action="store_true", help="""If set predicts""")
    parser.add_argument("-l", action="store_true", help="""Logs data on wandb""")
    parser.add_argument('-d', default='p', choices=['t', 'p', 'b'],
                        help="""Dataset specified (t: train, p: pred / test, b: both)""")
    args = parser.parse_args()

    bt = BevTraversability(args.l)

    if args.d == 't':
        TEST_DATASETS = ["train"]
    elif args.d == 'p':
        TEST_DATASETS = ["test"]
    else:
        TEST_DATASETS = ["train", "test"]

    # Setting training mode
    if args.t:
        bt.train(save_model=True)
    elif args.p:
        for TEST_DATASET in TEST_DATASETS:
            bt.predict(load_model=True, model_name=MODEL_NAME, save_pred=True, test_dataset=TEST_DATASET)
    else:
        raise ValueError(f"Unknown mode, please specify -t (for train mode), -p (for test mode)")
