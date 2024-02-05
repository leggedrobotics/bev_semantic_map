#!/usr/bin/env python

"""
Demo dataset for testing BEVNet.

Author: Robin Schmid
Date: Sep 2023
"""

import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from bevnet.dataset import normalize_img
from bevnet.cfg import DataParams, RunParams, ModelParams
from scipy.spatial.transform import Rotation as Rot

class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data: DataParams, cfg_run: RunParams, cfg_model: ModelParams):
        super(DemoDataset, self).__init__()
        self.cfg_data = cfg_data
        self.cfg_run = cfg_run
        self.cfg_model = cfg_model

        self.img_paths = sorted(glob.glob(os.path.join(self.cfg_data.data_dir, "img", "*")))
        self.pcd_paths = sorted(glob.glob(os.path.join(self.cfg_data.data_dir, "pcd", "*")))
        self.target_paths = sorted(glob.glob(os.path.join(self.cfg_data.data_dir, "bin_trav", "*")))

    def __len__(self):
        # return self.cfg_data.nr_data
        # return len(self.img_paths)
        if self.cfg_run.nr_data < 0 or self.cfg_data.mode != "train":
            if len(self.img_paths) > 0:
                return len(self.img_paths)
            else:
                return len(self.pcd_paths)
        else:
            return self.cfg_run.nr_data

    def get_image_data(self, idx, ):
        imgs = []
        img_plots = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for _ in range(self.cfg_data.nr_cameras):
            intrin = torch.Tensor(self.cfg_data.intrin).reshape(3, 3)

            if self.cfg_model.image_backbone == "skip":
                img = np.zeros((self.cfg_data.img_height, self.cfg_data.img_width, 3), dtype=np.uint8)
            else:
                # img = np.array(torch.load(self.img_paths[idx]).permute(1, 2, 0).cpu())
                img = cv2.imread(self.img_paths[idx])
            img_plot = normalize_img(img)  # Perform potential augmentation to plot the image

            # resize to fH, fW
            img = cv2.resize(img, (self.cfg_model.lift_splat_shoot_net.augmentation.fW, self.cfg_model.lift_splat_shoot_net.augmentation.fH))

            # perform some augmentation on the image / is now ignored
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)

            imgs.append(normalize_img(img))
            intrins.append(intrin)

            H_base_camera = torch.eye(4)  # 4d tensor for tf from base to camera frame
 
            H_base_camera[:3, :3] = torch.from_numpy(Rot.from_quat(self.cfg_data.rot_base_cam).as_matrix())
            H_base_camera[:3, 3] = torch.from_numpy(np.array(self.cfg_data.trans_base_cam))
            rots.append(H_base_camera[:3, :3])
            trans.append(H_base_camera[:3, 3])

            post_rots.append(post_rot)
            post_trans.append(post_tran)  # Rotation after performing augmentation
            img_plots.append(img_plot)  # Translation after performing augmentation

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
            torch.stack(img_plots),
        )

    def get_raw_pcd_data(self, idx):
        # H_pc_cam = [*self.cfg_data.trans_base_cam, *self.cfg_data.rot_base_cam]
        pcd_new = {}
        pcd_new["points"] = []
        for idx_pointcloud in range(self.cfg_data.nr_lidar_points_time):  # Only one lidar point for now
            # points_in_base_frame = torch.rand((self.cfg_data.nr_points, 3))  # Random points, uniformly between 0
            # and 1
            if idx + idx_pointcloud < len(self.pcd_paths):
                idx = idx + idx_pointcloud

            points_in_base_frame = torch.load(self.pcd_paths[idx])

            # points_in_cam_frame = self.project_pc(points_in_base_frame, H_pc_cam)

            pcd_new["points"].append(points_in_base_frame)  # Add points in base frame

        return pcd_new

    def project_pc(self, pc, pose):
        # Transform points to frame
        position = np.array(pose[:3])
        R = np.array(Rot.from_quat(pose[3:]).as_matrix())

        points_list = []
        for p in pc:
            p = np.matmul(R[:3, :3], np.array(p)) + position
            points_list.append(tuple(p))
        return torch.tensor(points_list, dtype=torch.float32)

    def get_dummy_target(self):
        target = torch.zeros(self.cfg_data.target_shape)
        # Fill left half with ones
        target[:, : int(self.cfg_data.target_shape[1] / 2), :] = 1
        return target

    def __getitem__(self, idx):  # Called when iterating over the dataset

        H_base_map = torch.eye(4)  # 4d tensor for tf from base to map frame, changing
        grid_map_resolution = torch.tensor([self.cfg_data.grid_map_resolution])

        # target, aux = torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512))    # Labels and aux labels in BEV space
        target, aux = (
            torch.zeros(self.cfg_data.target_shape),
            torch.zeros(self.cfg_data.aux_shape),
        )  # Labels and aux labels in BEV space

        if len(self.target_paths) > 0:
            target_np = torch.load(self.target_paths[idx])
            target = torch.from_numpy(target_np).unsqueeze(0)  # (1, 512, 512), for numpy arrays

        # target = torch.load(self.target_paths[idx]).unsqueeze(0)    # (1, 512, 512)

        # Get dummy image for debugging
        # target = self.get_dummy_target(
        # Save target image for debugging
        # target_out = target.permute(1, 2, 0).cpu().numpy()
        # target_out = (target_out * 255).astype(np.uint8)
        # cv2.imwrite(f"/home/rschmid/RosBags/bevnet/dummy/{idx}.jpg", target_out)

        imgs, rots, trans, intrins, post_rots, post_trans, img_plots = self.get_image_data(idx)

        pcd_new = self.get_raw_pcd_data(idx)

        return (
            imgs,
            rots,
            trans,
            intrins,
            post_rots,  # After performing augmentations
            post_trans,  # After performing augmentations
            target,
            aux,  # aux is ignored
            img_plots,  # img_plots is ignored
            grid_map_resolution,  # grid_map_resolution is ignored
            pcd_new,
        )


def collate_fn(batch):  # Prevents automatic data loading, performs operations over batches of data
    output_batch = []
    for i in range(len(batch[0])):

        if type(batch[0][i]) != dict:
            # iterate over tuple of data
            output_batch.append(torch.stack([item[i] for item in batch]))
        else:
            # dicts are raw pointclouds, only perform this for the pointclouds
            res = {}
            stacked_scans_ls = []
            stacked_scan_indexes = []
            for j in range(len(batch)):
                stacked_scans_ls.append(torch.cat(batch[j][i]["points"]))  # Concatenate all scans
                stacked_scan_indexes.append(
                    torch.tensor([scan.shape[0] for scan in batch[j][i]["points"]])
                )  # Get the index of all the scans

            res["points"] = torch.cat(stacked_scans_ls)
            res["scan"] = torch.cat(stacked_scan_indexes)
            res["batch"] = torch.stack(stacked_scan_indexes).sum(1)  # Get the number of points in each scan

            output_batch.append(res)

    return tuple(output_batch)


def get_bev_dataloader(mode="train", batch_size=1, shuffle=False):
    data_cfg = DataParams(mode=mode)
    run_cfg = RunParams()
    model_cfg = ModelParams()
    dataset = DemoDataset(data_cfg, run_cfg, model_cfg)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    return data_loader, data_cfg.data_dir


if __name__ == "__main__":
    data_loader = get_bev_dataloader()
    for j, batch in enumerate(data_loader):
        print(j)
