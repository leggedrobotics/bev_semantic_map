from perception_bev_learning.utils import load_yaml, load_pkl, get_H, get_H_h5py
from perception_bev_learning.utils import (
    normalize_img,
    img_transform,
    inv,
    get_gravity_aligned,
)
from perception_bev_learning import BEV_ROOT_DIR

import torch
from os.path import join
from PIL import Image
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import rotate, affine, center_crop
from dataclasses import asdict
from torch import from_numpy as fn
from torchvision.transforms.functional import pad, erase, to_tensor

import h5py
import random
from pytictac import ClassTimer, ClassContextTimer, cpu_accumulate_time, CpuTimer
import os
from typing import Dict, Any
from scipy import interpolate
from perception_bev_learning.dataset import BevDataset


class BevDatasetDepth(BevDataset):
    def __init__(
        self, cfg: Dict[str, Any], mode: str = None, deterministic_shuffle=None
    ):
        super().__init__(cfg, mode, deterministic_shuffle)
        self.depth_keys = cfg.depth_keys
        self.image_depth_dict = {
            self.image_keys[i]: self.depth_keys[i] for i in range(len(self.image_keys))
        }

    @cpu_accumulate_time
    def get_image_data(self, datum, H_sensor_gravity__map):
        imgs = []
        depths = []
        img_plots = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        # for h5py_camera_key, img_name in cameras:
        for img_key in self.image_keys:
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # TODO (Manthan) : Currently it is hard-coded. Should use dictionary
            camera_info_key = str(img_key) + "-camera_info"
            sk = datum["sequence_key"]

            depth_key = self.image_depth_dict[img_key]
            h5py_camera_info = self.h5py_handles[sk][sk][camera_info_key]
            h5py_image = self.h5py_handles[sk][sk][img_key]
            h5py_depth = self.h5py_handles[sk][sk][depth_key]

            idx = datum[img_key]
            idx_depth = datum[depth_key]

            arr = np.array(h5py_image[f"image"][idx])  # -{idx}
            img = Image.fromarray(arr[:, :, ::-1], mode="RGB")

            depth_arr = np.array(h5py_depth[f"data"][idx_depth]).squeeze(-1)

            depth_img = Image.fromarray(depth_arr, mode="F")

            intrin = (
                torch.tensor(h5py_camera_info["P"])
                .reshape(3, 4)[:3, :3]
                .type(torch.float32)
            )

            # # TODO something went wrong with some Camp Roberts DS so correcting here
            if intrin[0, 2] == 640:
                intrin[:2, :] *= 0.5

            # Placeholder for the real image
            # TODO remove img_plot
            img_plot = normalize_img(img)

            H_map__camera = get_H_h5py(
                t=h5py_image[f"tf_translation"][idx],
                q=h5py_image[f"tf_rotation_xyzw"][idx],
            )
            H_sensor_gravity__camera = H_sensor_gravity__map @ H_map__camera
            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            depth_img, _, _ = img_transform(
                depth_img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            imgs.append(normalize_img(img))
            depths.append(to_tensor(depth_img).squeeze(0))
            intrins.append(intrin)
            rots.append(H_sensor_gravity__camera[:3, :3])
            trans.append(H_sensor_gravity__camera[:3, 3])

            post_rots.append(post_rot)
            post_trans.append(post_tran)
            img_plots.append(img_plot)

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
            torch.stack(img_plots),
            torch.stack(depths),
        )

    @cpu_accumulate_time
    def __getitem__(self, index):
        """
        Returns a dictionary of items
        imgs -> 4 x 3 x H x W
        rots -> 4 x 3 x 3
        trans-> 4 x 3
        intrins -> 4 x 3 x 3
        post_rots -> 4 x 3 x 3
        post_trans -> 4 X 3
        target -> 2, 512, 512
        aux -> 6, 512, 512
        img_plots,
        gridmap_resolution,
        H_sg__map,
        pcd_new,
        gvom
        """
        index_new = self.index_mapping[index]

        datum = self.dataset_config[index_new]

        target, aux, H_sensor_gravity__map, grid_map_resolution = self.get_gridmap_data(
            datum
        )

        if self.cfg.return_image:
            (
                imgs,
                rots,
                trans,
                intrins,
                post_rots,
                post_trans,
                img_plots,
                depths,
            ) = self.get_image_data(datum, H_sensor_gravity__map)
        else:
            imgs, rots, trans, intrins, post_rots, post_trans, img_plots = tuple(
                [torch.tensor([[[0]]])] * 7
            )

        if self.cfg.return_n_pointclouds != 0:
            pcd_new = self.get_raw_pcd_data(datum, H_sensor_gravity__map)

        else:
            pcd_new = {}

        if self.cfg.return_gvom_cloud:
            gvom = self.get_gvom_data(datum, H_sensor_gravity__map)
        else:
            gvom = {}

        if self.ct_enabled:
            print(self._cct)

        if self.cfg.mode == "train" and self.cfg.augment_gridmap:
            aug_layer = [
                idx
                for idx, l in enumerate(self.cfg.aux_layers.values())
                if l.name == "elevation_raw"
            ][0]
            C, H, W = aux.shape
            i = torch.randint(0, H, size=(1,)).item()
            j = torch.randint(0, W, size=(1,)).item()
            h = torch.randint(10, 80, size=(1,)).item()
            w = torch.randint(10, 80, size=(1,)).item()

            if torch.rand(1) > 0.5:
                aux[aug_layer] = erase(aux[aug_layer], i, j, h, w, v=torch.nan)

        return {
            "imgs": imgs,
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "post_rots": post_rots,
            "post_trans": post_trans,
            "target": target,
            "aux": aux,
            "img_plots": img_plots,
            "gm_res": grid_map_resolution,
            "pcd": pcd_new,
            "gvom": gvom,
            "index": torch.tensor(index),
            "H_sg_map": H_sensor_gravity__map,
            "depths": depths,
        }
