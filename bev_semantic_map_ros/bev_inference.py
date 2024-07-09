#!/usr/bin/env python
import torch
import numpy as np
from os.path import join
from pathlib import Path
import pickle
import sys
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torchvision.transforms.functional import rotate, InterpolationMode, affine
from scipy.spatial.transform import Rotation as R
import math
import PIL
import time
import seaborn as sns
from pytictac import Timer
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F

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

sys.path.append("/home/rschmid/catkin_ws/src/bev_semantic_map")

# from perception_bev_learning.loss import LossManagerMulti
# from perception_bev_learning.utils import normalize_img, get_gravity_aligned

WEIGHTS_PATH = "/home/rschmid/git/bev_semantic_map/weights/2024_02_29_16_07_09_9.pth" # For testing
MODEL_NAME = "2024_03_08_10_22_10"

POS_WEIGHT = 0.2  # Num neg / num pos
VISU_TRAIN_EPOCHS = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "bevnet", "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import signal
import sys
from functools import partial
from typing import List

print("Finished imports")


def chop(t, dim=6):
    if len(t.shape) == 2:
        return t[dim:-dim, dim:-dim]
    elif len(t.shape) == 3:
        return t[:, dim:-dim, dim:-dim]
    elif len(t.shape) == 4:
        return t[:, :, dim:-dim, dim:-dim]


class BevInference:
    def __init__(self, wandb_logging=False, img_backbone=False, pcd_backbone=False):
        # Loading params and the model

        self._model_cfg = ModelParams()
        self._run_cfg = RunParams()
        self._data_cfg = DataParams()

        self.weights_path = WEIGHTS_PATH

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

        self._model.load_state_dict(self.weights_path, map_location=torch.device(DEVICE), strict=True)

        # # For now, just use the hardcoded checkpoint path for loading the model
        # self.img_count = 0
        # # Register the signal handler for Ctrl+C
        # signal.signal(signal.SIGINT, partial(self.signal_handler, self))

        # # checkpoint = "/data_large/manthan/crf_trained_models/256_multi_consistency_32k/epoch=24-step=32000---last.ckpt"
        # if checkpoint is None:
        #     checkpoints = [str(s) for s in Path(weight_dir).rglob("*.ckpt")]
        #     checkpoints.sort()
        #     checkpoint = checkpoints[-1]

        # print(f"Loading the following checkpoint: {checkpoint}")

        # ckpt = torch.load(checkpoint)
        # print(f"loaded checkpoint")
        # cfg = OmegaConf.load(join(weight_dir, "hydra/.hydra/config.yaml"))
        # self._cfg = cfg

        # cfg_model = self._cfg.model.network
        # # Initalize Model
        # model = hydra.utils.instantiate(cfg_model)

        # loss_dict = {
        #     "_target_": "perception_bev_learning.loss.LossManagerMulti",
        #     "cfg_layers": cfg.model.target_layers,
        # }

        # loss_config = OmegaConf.create(loss_dict)
        # self._loss_manager = hydra.utils.instantiate(loss_config)
        # print(model)

        # state_dict = {
        #     k[len("net.") :] if k.startswith("net.") else k: v
        #     for k, v in ckpt["state_dict"].items()
        #     if "net" in k
        # }

        # model.load_state_dict(state_dict)
        # model.to(self.device)

        # # If no specific name is given load the latest model
        # if model_name is None:
        #     all_directories = [d for d in os.listdir(DATA_PATH) if
        #                         os.path.isdir(os.path.join(DATA_PATH, d))]
        #     sorted_directories = sorted(all_directories)
        #     model_name = sorted_directories[-1]

        # try:
        #     weights_dir = os.path.join(DATA_PATH, model_name, "weights")
        #     all_weights = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]

        #     # Function to extract all numbers from the filename and return them as a tuple of integers
        #     def extract_numbers(s):
        #         numbers = re.findall(r'\d+', s)
        #         return tuple(int(number) for number in numbers)

        #     # Sort files based on the numerical parts extracted
        #     sorted_weights = sorted(all_weights, key=extract_numbers)

        #     # Select the last file after sorting
        #     latest_weight_file = sorted_weights[-1]

        #     print(f"Using model {latest_weight_file}")

        #     self._model.load_state_dict(
        #         torch.load(
        #             os.path.join(weights_dir, latest_weight_file),
        #             map_location=torch.device(DEVICE)
        #         ), strict=True
        #     )
        # except:
        #     ValueError("This model configuration does not exist!")

        # # Set the model to evaluation mode
        # self._model.eval()

        # self.scale_target = nn.ParameterDict()
        # self.scale_aux = nn.ParameterDict()

        # for gridmap_key in self._cfg.model.metrics.target_layers.keys():
        #     scale_target = torch.tensor(
        #         [
        #             l.scale
        #             for l in self._cfg.model.metrics.target_layers[
        #                 gridmap_key
        #             ].layers.values()
        #         ]                                                                                                                                                                                                                                                                                                           
        #     )[None, :, None, None]
        #     self.scale_target[gridmap_key] = nn.Parameter(
        #         scale_target, requires_grad=False
        #     )

        #     scale_aux = torch.tensor(
        #         [
        #             l.scale
        #             for l in self._cfg.model.metrics.aux_layers[
        #                 gridmap_key
        #             ].layers.values()
        #         ]
        #     )[None, :, None, None]
        #     self.scale_aux[gridmap_key] = nn.Parameter(scale_aux, requires_grad=False)

        # # self.aux_clip_min = torch.zeros(
        # #     (len(self._cfg.datamodule.dataset.aux_layers))
        # # ).to(self.device)
        # # self.aux_clip_max = torch.ones_like(self.aux_clip_min)
        # # self.aux_scale = torch.ones_like(self.aux_clip_min)

        # # for j, target in enumerate(self._cfg.datamodule.dataset.aux_layers.values()):
        # #     self.aux_scale[j] = target.scale
        # #     self.aux_clip_min[j] = target.clip_min
        # #     self.aux_clip_max[j] = target.clip_max

        # self.inf_time_list = []
        # self.pcd_size_list = []
        # print("-------- BEV Inference Python INITIALIZED --------")

    # Instance method to handle Ctrl+C (KeyboardInterrupt) signal
    def signal_handler(self, instance, sig, frame):
        # Perform cleanup actions here (if any)
        print("Received Ctrl+C, performing cleanup...")
        sys.exit(0)

    def get_gravity_aligned_py(self, H_f__map):
        print("Entered python")
        print(f"Python H is {H_f__map}")
        ypr = R.from_matrix(np.asarray(H_f__map)[:3, :3]).as_euler(
            seq="zyx", degrees=False
        )
        print(f"YAW, Pitch, Roll python is {ypr}")
        H_g__map = H_f__map.copy()
        H_delta = np.eye(4)

        ypr[0] = 0
        H_delta[:3, :3] = R.from_euler(seq="zyx", angles=ypr, degrees=False).as_matrix()
        H_g__map = np.linalg.inv(H_delta) @ H_g__map

        return H_g__map

    def get_ypr_py(self, H):
        print("Entered ypr py")
        ypr = R.from_matrix(np.asarray(H)[:3, :3]).as_euler(seq="zyx", degrees=False)
        return np.array(ypr)

    def infer_python(
        self,
        imgList: List[np.ndarray],
        imgRotList: List[np.ndarray],
        imgTransList: List[np.ndarray],
        imgIntrinsics: List[np.ndarray],
        cloud,
        cloudTF,
        rawEleMap: np.ndarray,
        yaw,
        shift,
        T_sensor_origin__map,
        T_gravity__map,
        wheel_risk_micro,
        elevation_micro,
        wheel_risk_short,
        elevation_short,
    ):
        # def infer_python(self, imgList: List[np.ndarray], cloud, cloudTF):
        start_time = time.time()

        print("Performing inference")

        # T_python_gravity = get_gravity_aligned(torch.as_tensor(T_sensor_origin__map, dtype=torch.float))
        # print("GRAVITY ALIGNED")
        # print(T_python_gravity)
        # print(T_gravity__map)

        imgs = []
        intrins = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rot = torch.eye(3, dtype=torch.float32)

        for i in range(len(imgList)):
            curr_img = np.asarray(imgList[i], dtype=np.uint8).transpose([1, 2, 0])
            print(curr_img.shape)  # 256x384x3
            rots.append(torch.as_tensor(imgRotList[i], dtype=torch.float32))
            trans.append(torch.as_tensor(imgTransList[i], dtype=torch.float32))
            intrins.append(torch.as_tensor(imgIntrinsics[i], dtype=torch.float32))
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            imgs.append(normalize_img(curr_img))

        imgs = torch.stack(imgs)[None].to(self.device)
        rots = torch.stack(rots)[None].to(self.device)
        trans = torch.stack(trans)[None].to(self.device)
        intrins = torch.stack(intrins)[None].to(self.device)
        post_rots = torch.stack(post_rots)[None].to(self.device)
        post_trans = torch.stack(post_trans)[None].to(self.device)
        # aux = aux[None].to(self.device)

        # Point Cloud data
        cloud = torch.from_numpy(np.asarray(cloud).astype(np.float32)).to(self.device)
        cloud = torch.cat(
            [cloud, torch.ones((cloud.shape[0], 1), device=cloud.device)], dim=1
        )
        H_sensor_gravity__base_link = torch.from_numpy(
            np.asarray(cloudTF).astype(np.float32)
        ).to(self.device)
        sensor_gravity_points = (
            H_sensor_gravity__base_link.to(self.device) @ cloud.T
        ).T
        sensor_gravity_points = sensor_gravity_points[:, :3]

        pcd_new = {}
        stacked_scan_indexes = []
        stacked_scan_indexes = torch.tensor(
            [scan.shape[0] for scan in [sensor_gravity_points]]
        )

        pcd_new["points"] = sensor_gravity_points.to(self.device).contiguous()
        pcd_new["scan"] = stacked_scan_indexes.to(self.device)
        pcd_new["batch"] = stacked_scan_indexes.to(self.device)
        aux = torch.zeros(
            (1, len(self._cfg.datamodule.dataset.aux_layers["short"].layers), 256, 256),
            device=self.device,
            dtype=torch.float32,
        )  # dummy aux layer

        self.pcd_size_list.append(pcd_new["points"].shape[0])

        # # GVOM Data
        # gvom = torch.from_numpy(np.asarray(gvom).astype(np.float32)).to(self.device)
        # gvom = torch.cat(
        #     [gvom, torch.ones((gvom.shape[0], 1), device=gvom.device)], dim=1
        # )
        # H_sensor_gravity__map = torch.from_numpy(
        #     np.asarray(T_gravity__map).astype(np.float32)
        # ).to(self.device)
        # sensor_gravity_points = (H_sensor_gravity__map.to(self.device) @ gvom.T).T
        # sensor_gravity_points = sensor_gravity_points[:, :3]

        # gvom_new = {}
        # stacked_scan_indexes = []
        # stacked_scan_indexes = torch.tensor(
        #     [scan.shape[0] for scan in [sensor_gravity_points]]
        # )

        # gvom_new["points"] = sensor_gravity_points.to(self.device).contiguous()
        # gvom_new["scan"] = stacked_scan_indexes.to(self.device)
        # gvom_new["batch"] = stacked_scan_indexes.to(self.device)

        # Elevation Processing
        # rawEleMap = rawEleMap[::-1, ::-1].transpose().astype(np.float32)
        rawEleMap = np.flip(rawEleMap, 0)
        rawEleMap = np.flip(rawEleMap, 1)

        ele_map = torch.from_numpy(
            np.ascontiguousarray(rawEleMap, dtype=np.float32)
        ).to(self.device)

        H_c, W_c = int(ele_map.shape[0] / 2), int(ele_map.shape[1] / 2)
        sh = [shift[1], shift[0]]

        # print(f"Elevation Map shape is {ele_map.shape}")
        # print(f"Shift is {sh}")
        yaw = yaw * (180 / math.pi)
        # print(f"Yaw is {yaw}")
        # print(f"Center is {H_c} , {W_c}")

        grid_map_data_rotated = affine(
            ele_map[None],
            angle=-yaw,
            translate=sh,
            scale=1,
            shear=0,
            center=(H_c, W_c),
            fill=torch.nan,
        )

        grid_map_data_rotated *= 0.05
        grid_map_data_rotated = grid_map_data_rotated.clip(-1, 1)

        grid_map_data_rotated = nn.functional.interpolate(
            grid_map_data_rotated[None, ...], scale_factor=0.625
        )[0]
        grid_map_data_rotated = pad(grid_map_data_rotated, 3)

        # print(f"The shape of rotated tensor is {grid_map_data_rotated.shape}")

        # aux[0, 1, ...] = grid_map_data_rotated  # Raw elevation is at index 1

        # # pcd_plot = np.asarray(sensor_gravity_points[:, :2].cpu() / 0.2, dtype=np.int32)

        # # Plot Aux Layer
        # cmap_elevation = sns.color_palette("viridis", as_cmap=True)
        # cmap_elevation.set_bad(color="black")
        # # plt.imshow(grid_map_data_rotated[0].cpu().numpy(), cmap=cmap_elevation, vmin=-20, vmax=20)
        # # plt.savefig(f'/home/jonfrey/images_bev/trial{self.img_count}.png')

        # image_array = grid_map_data_rotated[0].cpu().numpy()  # Assuming this is your NumPy array
        # image_array = (image_array * 255).astype(np.uint8)  # Scale to 0-255 range if necessary
        # colored_image = cmap_elevation(image_array)
        # colored_image = (colored_image * 255).astype(np.uint8)  # Scale to 0-255 range

        # # Save the image using OpenCV
        # image_path = f'/home/jonfrey/images_bev2/reliable_shift{self.img_count}.png'

        # # for point in pcd_plot:
        # #     x, y = int(point[0]), int(point[1])
        # #     color = (255, 0, 0)  # White color for points
        # #     cv2.circle(colored_image, (256-y, 256-x), 3, color)

        # cv2.imwrite(image_path, colored_image)

        # aux *= self.aux_scale[:, None, None]
        # aux = aux.clip(
        #     self.aux_clip_min[:, None, None], self.aux_clip_max[:, None, None]
        # )

        before_inf_time = time.time()
        print("Starting inference")
        with torch.no_grad():
            with Timer("model"):
                pred = self._model(
                    imgs=imgs,
                    rots=rots,
                    trans=trans,
                    intrins=intrins,
                    post_rots=post_rots,
                    post_trans=post_trans,
                    pcd=pcd_new,
                    aux=aux,
                    H_sg_map=H_sensor_gravity__map,
                    batch_idx=self.img_count,
                )

            after_inf_time = time.time()

            # loss, final_pred = self._loss_manager.compute(
            #     pred, target=None, compute_loss=False
            # )

            pred["short"]["cost"] = nn.functional.sigmoid(pred["short"]["cost"])
            pred["micro"]["wheel_risk"] = nn.functional.sigmoid(
                pred["micro"]["wheel_risk"]
            )

            final_pred_short = torch.cat(
                (pred["short"]["cost"], pred["short"]["elevation"]), dim=1
            )
            final_pred_short = chop(final_pred_short, dim=3)[0]

            final_pred_micro = torch.cat(
                (pred["micro"]["wheel_risk"], pred["micro"]["elevation"]), dim=1
            )
            final_pred_micro = chop(final_pred_micro, dim=6)[0]

            # Rotate back the predictions
            yaw = R.from_matrix(np.asarray(T_gravity__map)[:3, :3]).as_euler(
                seq="zyx", degrees=True
            )[0]

            trial_time = time.time()

            pred_rotated_micro = affine(
                final_pred_micro[None],
                angle=yaw,
                translate=[0, 0],
                scale=1,
                shear=0,
                fill=torch.nan,
            )[0]
            pred_rotated_short = affine(
                final_pred_short[None],
                angle=yaw,
                translate=[0, 0],
                scale=1,
                shear=0,
                fill=torch.nan,
            )[0]

            # on GPu
            trial_time2 = time.time()
            # Rescale the predictions

            for i, layer in enumerate(
                self._cfg.datamodule.dataset.target_layers["short"].layers.values()
            ):
                pred_rotated_short[i] *= 1 / layer.scale
            for i, layer in enumerate(
                self._cfg.datamodule.dataset.target_layers["micro"].layers.values()
            ):
                pred_rotated_micro[i] *= 1 / layer.scale

            H_map__gravity = torch.inverse(
                torch.from_numpy(T_gravity__map).to(self.device)
            )
            # Obtain elevation in the map frame
            pred_rotated_micro[1] += H_map__gravity[2, 3]
            pred_rotated_short[1] += H_map__gravity[2, 3]

            # Upsample the short predictions
            pred_rotated_short = F.interpolate(
                pred_rotated_short[None],
                size=(400, 400),
                mode="bilinear",
                align_corners=True,
            )[0]
            pred_rotated_micro = pred_rotated_micro.cpu().numpy()
            pred_rotated_short = pred_rotated_short.cpu().numpy()

            pred_rotated_micro[0] = np.flip(pred_rotated_micro[0], 0)
            pred_rotated_micro[0] = np.flip(pred_rotated_micro[0], 1)
            pred_rotated_micro[1] = np.flip(pred_rotated_micro[1], 0)
            pred_rotated_micro[1] = np.flip(pred_rotated_micro[1], 1)

            pred_rotated_short[0] = np.flip(pred_rotated_short[0], 0)
            pred_rotated_short[0] = np.flip(pred_rotated_short[0], 1)
            pred_rotated_short[1] = np.flip(pred_rotated_short[1], 0)
            pred_rotated_short[1] = np.flip(pred_rotated_short[1], 1)

            # pred_rotated[0] = pred_rotated[0] > 0.6
            wheel_risk_micro[...] = np.array(pred_rotated_micro[0], dtype=np.float32)
            elevation_micro[...] = np.array(pred_rotated_micro[1], dtype=np.float32)

            wheel_risk_short[...] = np.array(pred_rotated_short[0], dtype=np.float32)
            elevation_short[...] = np.array(pred_rotated_short[1], dtype=np.float32)

            # final_pred = final_pred.cpu().numpy()
            # pred_mask = np.ones_like(final_pred[0]).astype(bool)
            # plot_pred = np.ma.masked_where(~pred_mask, final_pred[0])

            # cmap_traversability = sns.color_palette("RdYlBu_r", as_cmap=True)
            # plot_pred = (plot_pred * 255).astype(np.uint8)  # Scale to 0-255 range if necessary
            # colored_image = cmap_traversability(plot_pred)
            # colored_image = (colored_image * 255).astype(np.uint8)  # Scale to 0-255 range
            # # Save the image using OpenCV
            # image_path = f'/home/jonfrey/images_bev/pred_wheelrisk{self.img_count}.png'
            # cv2.imwrite(image_path, colored_image)
            # # plt.imshow(final_pred, cmap=cmap_traversability, vmin=0, vmax=1)
            # # plt.savefig(f'/home/jonfrey/images_bev/pred_wheelrisk{self.img_count}.png')

        print("Finished inference")
        end_time = time.time()
        elapsed_time = end_time - start_time

        # print(f"Python preprocessing Time: {before_inf_time - start_time} seconds")
        # print(f"Python inference Time: {after_inf_time - before_inf_time} seconds")
        # print(f"Python postprocessing affine Time: {trial_time2 - trial_time} seconds")
        # print(f"Python postprocessing Time: {end_time - after_inf_time} seconds")
        # print(f"Python Loop Time: {elapsed_time} seconds")
        self.img_count = self.img_count + 1
        print(f"No of Images: {self.img_count}")

        if self.img_count > 100:
            self.inf_time_list.append(after_inf_time - before_inf_time)
            print(f"Mean Python inference time is {np.mean(self.inf_time_list)}")
            print(f"Std Dev Python inference time is {np.std(self.inf_time_list)}")
            print(f"Mean PCD size is {np.mean(self.pcd_size_list)}")


if __name__ == "__main__":
    print("Init")
    bev_inference = BevInference()
