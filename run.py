#!/usr/bin/env python

"""
BEVnet for safe and unsafe traversability prediction.

Author: Robin Schmid
Date: Sep 2023
"""

import re
import os
import cv2
import time
import json
from tqdm import tqdm
import numpy as np
import torch
import wandb
import argparse
from dataclasses import asdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bevnet.cfg import ModelParams, RunParams, DataParams
from bevnet.network.bev_net import BevNet
from bevnet.dataset import get_bev_dataloader
from bevnet.utils import Timer, compute_evaluation

# Global settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=200)
# matplotlib.use('Agg')   # To avoid using X server and run it in the background

MODEL_NAME = None
MODEL_NAME = "2024_03_08_10_22_10"
# MODEL_NAME = "2024_02_08_15_33_46"    # Specify a specific model
# MODEL_NAME = "2024_02_19_09_22_53"

POS_WEIGHT = 0.2  # Num neg / num pos
VISU_TRAIN_EPOCHS = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "bevnet", "data")


class BevTraversability:
    def __init__(self, wandb_logging=False, img_backbone=False, pcd_backbone=False):
        self._model_cfg = ModelParams()
        self._run_cfg = RunParams()
        self._data_cfg = DataParams()

        if img_backbone:
            self._model_cfg.image_backbone = "lift_splat_shoot_net"
        if pcd_backbone:
            self._model_cfg.pointcloud_backbone = "point_pillars"

        self._model = BevNet(self._model_cfg)
        self._model.cuda()

        self.wandb_logging = wandb_logging

        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._run_cfg.lr)

        self._loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([POS_WEIGHT, 1-POS_WEIGHT]), ignore_index=-1)
        self._loss.cuda()
        self._loss_mean = torch.tensor(0.0)

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
            self._run_cfg.log_name = model_name + "_img_" + self._model_cfg.image_backbone + "_pcd_" + self._model_cfg.pointcloud_backbone
            json.dump(asdict(self._run_cfg), json_file, indent=2)
        with open(os.path.join(DATA_PATH, model_name, "config", "data_params.json"), 'w') as json_file:
            json.dump(asdict(self._data_cfg), json_file, indent=2)

        data_loader, _ = get_bev_dataloader(mode="train", batch_size=self._run_cfg.training_batch_size, shuffle=True, model_cfg=self._model_cfg)

        if self.wandb_logging:
            wandb.init(project=self._run_cfg.log_name)

        for i in tqdm(range(self._run_cfg.epochs), desc="Epochs"):
            try:
                for _, batch in enumerate(tqdm(data_loader, desc=f"Epoch {i+1} / {self._run_cfg.epochs} | Loss {self._loss_mean.item():.9f} | Batches")):
                    imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
                    pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = (
                        pcd_new["points"].cuda(),
                        pcd_new["batch"].cuda(),
                        pcd_new["scan"].cuda(),
                    )

                    # img_to_visualize = imgs[0].squeeze().cpu().numpy()  # Convert the first image tensor to numpy
                    # plt.imshow(np.transpose(img_to_visualize, (1, 2, 0))* 255)  # Transpose the image for correct display
                    # plt.title("First Image in Batch")
                    # plt.show()

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
                    self._loss_mean = self._loss(pred, target.long().cuda()[:, 0, :, :])

                    if self.wandb_logging:
                        wandb.log({"train_loss": self._loss_mean.item()})

                    # Backward pass
                    self._optimizer.zero_grad()
                    self._loss_mean.backward()
                    self._optimizer.step()
            except Exception as e:
                print(f"Error in epoch {i}: {e}")

            if VISU_TRAIN_EPOCHS:
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
                # and i % 10 == 0:
                print(f"Saving model for epoch {i} as {model_name}")
                torch.save(self._model.state_dict(),
                           os.path.join(DATA_PATH, model_name, "weights", f"{model_name}_{i}.pth"))

    def predict(self, load_model=True, model_name=None, save_pred=False, test_dataset=None):
        if load_model:
            self._model.to(DEVICE)

            # If no specific name is given load the latest model
            if model_name is None:
                all_directories = [d for d in os.listdir(DATA_PATH) if
                                   os.path.isdir(os.path.join(DATA_PATH, d))]
                sorted_directories = sorted(all_directories)
                model_name = sorted_directories[-1]

            try:
                weights_dir = os.path.join(DATA_PATH, model_name, "weights")
                all_weights = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]

                # Function to extract all numbers from the filename and return them as a tuple of integers
                def extract_numbers(s):
                    numbers = re.findall(r'\d+', s)
                    return tuple(int(number) for number in numbers)

                # Sort files based on the numerical parts extracted
                sorted_weights = sorted(all_weights, key=extract_numbers)

                # Select the last file after sorting
                latest_weight_file = sorted_weights[-1]

                print(f"Using model {latest_weight_file}")

                self._model.load_state_dict(
                    torch.load(
                        os.path.join(weights_dir, latest_weight_file),
                        map_location=torch.device(DEVICE)
                    ), strict=True
                )
            except:
                ValueError("This model configuration does not exist!")

            if not os.path.exists(os.path.join(DATA_PATH, model_name, f"pred_{test_dataset}")):
                os.makedirs(os.path.join(DATA_PATH, model_name, f"pred_{test_dataset}"))

            # Set the model to evaluation mode
            self._model.eval()

        data_loader, self._data_cfg.data_dir = get_bev_dataloader(mode=test_dataset, batch_size=1, model_cfg=self._model_cfg)
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

    def evaluate(self, test_dataset="test"):
        pred_path = os.path.join(DATA_PATH, MODEL_NAME, f"pred_{test_dataset}")
        gt_path = os.path.join(self._data_cfg.data_dir, "bin_trav_filtered")

        fig_path = os.path.join(DATA_PATH, MODEL_NAME, "eval_fig")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        print(pred_path)
        print(gt_path)
                                 
        compute_evaluation(gt_path=gt_path,
                       pred_path=pred_path,
                       fig_path=fig_path,
                       model_name=pred_path)


if __name__ == "__main__":
    # Passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true", help="""If set trains""")
    parser.add_argument("-p", action="store_true", help="""If set predicts""")
    parser.add_argument("-e", action="store_true", help="""If set evaluates""")
    parser.add_argument("-l", action="store_true", help="""Logs data on wandb""")
    parser.add_argument('-d', default='p', choices=['t', 'p', 'b'],
                        help="""Dataset specified (t: train, p: pred / test, b: both)""")
    parser.add_argument('--img', action="store_true", help="""Image backbone true""")
    parser.add_argument('--pcd', action="store_true", help="""Pointcloud backbone true""")
    args = parser.parse_args()

    bt = BevTraversability(args.l, args.img, args.pcd)

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
    elif args.e:
        for TEST_DATASET in TEST_DATASETS:
            bt.evaluate(test_dataset=TEST_DATASET)
    else:
        raise ValueError(f"Unknown mode, please specify -t (for train mode), -p (for test mode), -e (for evaluation mode) or -b (for both modes).")
