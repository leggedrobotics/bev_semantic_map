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
from torchvision.transforms.functional import pad, erase

import h5py
import random
from pytictac import ClassTimer, ClassContextTimer, cpu_accumulate_time, CpuTimer, Timer
import os
from typing import Dict, Any
from scipy import interpolate
import open3d as o3d


class BevDataset(torch.utils.data.Dataset):
    def __init__(
        self, cfg: Dict[str, Any], mode: str = None, deterministic_shuffle=None
    ):
        self.cfg = cfg

        if mode is not None:
            self.mode = mode
        else:
            self.mode = str(self.cfg.mode)
        if deterministic_shuffle is not None:
            self.cfg.deterministic_shuffle = deterministic_shuffle

        self.image_keys = cfg.image_keys
        self.pointcloud_key = cfg.pointcloud_key
        self.gvom_key = cfg.gvom_key

        self.dataset_config = load_pkl(join(cfg.root_dir, cfg.cfg_file))
        self.dataset_config = [d for d in self.dataset_config if d["mode"] == self.mode]

        # Get all sequences in the current dataset
        seq = [s["sequence_key"] for s in self.dataset_config]
        seq = np.unique(np.array(seq)).tolist()

        print(f"Opening in mode {self.mode} the following h5py files: ", seq)
        print(f"Using {cfg.root_dir} to load the hdf5 files")
        self.h5py_handles = {
            s: h5py.File(join(cfg.root_dir, s + ".h5py"), "r") for s in seq
        }

        self.length = len(self.dataset_config)
        self.setup_target_clip_scale()

        self.index_mapping = np.arange(0, self.length).astype(np.int32).tolist()

        if self.cfg.deterministic_shuffle:
            random.shuffle(self.index_mapping)

        self._cctk = 0
        self._cct = ClassTimer(
            objects=[self], names=["DataLoader"], enabled=cfg.enable_profiling
        )

    def __len__(self):
        return self.length

    @cpu_accumulate_time
    def setup_target_clip_scale(self):
        self.gridmap_topics = np.unique(
            [l.gridmap_topic for l in self.cfg.target_layers.values()]
            + [l.gridmap_topic for l in self.cfg.aux_layers.values()]
        )

        self.target_clip_min = torch.zeros((len(self.cfg.target_layers)))
        self.target_clip_max = torch.ones_like(self.target_clip_min)
        self.target_scale = torch.ones_like(self.target_clip_min)

        for j, target in enumerate(self.cfg.target_layers.values()):
            self.target_scale[j] = target.scale
            self.target_clip_min[j] = target.clip_min
            self.target_clip_max[j] = target.clip_max

        self.aux_clip_min = torch.zeros((len(self.cfg.aux_layers)))
        self.aux_clip_max = torch.ones_like(self.aux_clip_min)
        self.aux_scale = torch.ones_like(self.aux_clip_min)

        for j, target in enumerate(self.cfg.aux_layers.values()):
            self.aux_scale[j] = target.scale
            self.aux_clip_min[j] = target.clip_min
            self.aux_clip_max[j] = target.clip_max

        dim = tuple(self.cfg.target_shape)
        self.target_shape = (len(self.cfg.target_layers),) + dim
        self.aux_shape = (len(self.cfg.aux_layers),) + dim

    @cpu_accumulate_time
    def sample_augmentation(self):
        H, W = self.cfg.img_augmentation.H, self.cfg.img_augmentation.W
        fH, fW = self.cfg.img_augmentation.fH, self.cfg.img_augmentation.fW
        if self.mode == "train":
            resize = np.random.uniform(*self.cfg.img_augmentation.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.cfg.img_augmentation.bot_pct_lim))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.cfg.img_augmentation.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.cfg.img_augmentation.rot_lim)
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.cfg.img_augmentation.bot_pct_lim)) * newH) - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    @cpu_accumulate_time
    def get_image_data(self, datum, H_sensor_gravity__map):
        imgs = []
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

            h5py_camera_info = self.h5py_handles[sk][sk][camera_info_key]
            h5py_image = self.h5py_handles[sk][sk][img_key]

            idx = datum[img_key]
            arr = np.array(h5py_image[f"image"][idx])  # -{idx}
            img = Image.fromarray(arr[:, :, ::-1], mode="RGB")

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
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
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
        )

    # It is assumed that the GT labels contain the -50,50 pcd data
    @cpu_accumulate_time
    def get_raw_pcd_data(self, datum, H_sensor_gravity__map):
        pcd_new = {}
        pcd_new["points"] = []
        dist_threshold = self.cfg.pointcloud_merge_threshold

        idx_pointcloud = datum[self.pointcloud_key][-1]  # Most recent Cloud
        sk = datum["sequence_key"]
        h5py_pointcloud = self.h5py_handles[sk][sk][self.pointcloud_key]
        H_map__base_link = get_H_h5py(
            t=h5py_pointcloud[f"tf_translation"][idx_pointcloud],
            q=h5py_pointcloud[f"tf_rotation_xyzw"][idx_pointcloud],
        )
        valid_point = np.array(h5py_pointcloud[f"valid"][idx_pointcloud]).sum()
        x = h5py_pointcloud[f"x"][idx_pointcloud][:valid_point]
        y = h5py_pointcloud[f"y"][idx_pointcloud][:valid_point]
        z = h5py_pointcloud[f"z"][idx_pointcloud][:valid_point]
        points = fn(np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)).type(
            torch.float32
        )
        H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
        sensor_gravity_points = (H_sensor_gravity__base_link @ points.T).T
        sensor_gravity_points = sensor_gravity_points[:, :3]

        # # TODO: Option to concatenate intensity
        # intensity = h5py_pointcloud[f"intensity"][idx_pointcloud][:valid_point]
        # intensity = fn(intensity).type(torch.float32)
        # sensor_gravity_points = torch.stack(
        #     (sensor_gravity_points, intensity), dim=1
        # )
        ts_anchor = (
            h5py_pointcloud[f"header_stamp_secs"][idx_pointcloud]
            + h5py_pointcloud["header_stamp_nsecs"][idx_pointcloud] * 10**-9
        )

        pcd_new["points"].append(sensor_gravity_points)
        prev_H = H_map__base_link  # Initialize with the current H matrix
        n_clouds = 1
        # print(f"New Sample")
        # Take the most recent Pointclouds if they satisfy the distance threshold of 0.5 meters
        for idx_pointcloud in datum[self.pointcloud_key][50:]:
            # ts_curr = (
            #     h5py_pointcloud[f"header_stamp_secs"][idx_pointcloud]
            #     + h5py_pointcloud["header_stamp_nsecs"][idx_pointcloud] * 10**-9
            # )
            # print(ts_anchor - ts_curr)
            if n_clouds >= self.cfg.return_n_pointclouds:
                break

            sk = datum["sequence_key"]
            h5py_pointcloud = self.h5py_handles[sk][sk][self.pointcloud_key]
            H_map__base_link = get_H_h5py(
                t=h5py_pointcloud[f"tf_translation"][idx_pointcloud],
                q=h5py_pointcloud[f"tf_rotation_xyzw"][idx_pointcloud],
            )
            valid_point = np.array(h5py_pointcloud[f"valid"][idx_pointcloud]).sum()
            x = h5py_pointcloud[f"x"][idx_pointcloud][:valid_point]
            y = h5py_pointcloud[f"y"][idx_pointcloud][:valid_point]
            z = h5py_pointcloud[f"z"][idx_pointcloud][:valid_point]
            points = fn(np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)).type(
                torch.float32
            )
            H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
            dist = torch.norm(prev_H[:3, 3] - H_map__base_link[:3, 3])
            if dist > dist_threshold:
                sensor_gravity_points = (H_sensor_gravity__base_link @ points.T).T
                sensor_gravity_points = sensor_gravity_points[:, :3]
                pcd_new["points"].append(sensor_gravity_points)

                prev_H = H_map__base_link
                n_clouds += 1

        # Merge the clouds to a tensor and put them in a list to make compatible with varying concats
        pcd_new["points"] = torch.cat(pcd_new["points"])
        # with Timer("o3d voxel downsample"):
        if self.cfg.voxel_downsample:
            o3d_pc = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.array(pcd_new["points"]))
            )
            voxel_size = self.cfg.voxel_downsample_size
            downsampled_o3d_pc = o3d_pc.voxel_down_sample(voxel_size)
            np_pc = np.asarray(downsampled_o3d_pc.points)
            pcd_new["points"] = [torch.tensor(np_pc, dtype=torch.float32)]
        else:
            pcd_new["points"] = [pcd_new["points"]]

        return pcd_new

    @cpu_accumulate_time
    def get_gvom_data(self, datum, H_sensor_gravity__map):
        pcd_new = {}
        pcd_new["points"] = []
        idx_gvomcloud = datum[self.gvom_key]

        sk = datum["sequence_key"]
        h5py_pointcloud = self.h5py_handles[sk][sk][self.gvom_key]
        H_map__base_link = get_H_h5py(
            t=h5py_pointcloud[f"tf_translation"][idx_gvomcloud],
            q=h5py_pointcloud[f"tf_rotation_xyzw"][idx_gvomcloud],
        )
        valid_point = np.array(h5py_pointcloud[f"valid"][idx_gvomcloud]).sum()

        x = h5py_pointcloud[f"x"][idx_gvomcloud][:valid_point]
        y = h5py_pointcloud[f"y"][idx_gvomcloud][:valid_point]
        z = h5py_pointcloud[f"z"][idx_gvomcloud][:valid_point]
        points = fn(np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)).type(
            torch.float32
        )
        H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
        sensor_gravity_points = (H_sensor_gravity__base_link @ points.T).T
        sensor_gravity_points = sensor_gravity_points[:, :3]

        # # TODO (Add all prob fields)
        # prob_keys = [x for x in h5py_pointcloud.keys() if "prob" in x]
        # output_probs = []
        # for key in prob_keys:
        #     output_probs.append(h5py_pointcloud[key][idx_gvomcloud][:valid_point])

        # stacked_probs = torch.from_numpy(np.stack(output_probs).transpose()).type(
        #     torch.float32
        # )

        # sensor_gravity_points = torch.cat([sensor_gravity_points, stacked_probs], dim=1)
        pcd_new["points"].append(sensor_gravity_points)

        return pcd_new

    @cpu_accumulate_time
    def get_gridmap_data(self, datum):
        target = torch.zeros(self.target_shape, dtype=torch.float32)
        aux = torch.zeros(self.aux_shape, dtype=torch.float32)

        # Iterate over all gridmap topics that are needed
        for gridmap_key in self.gridmap_topics:
            gm_idx = datum[gridmap_key]

            sk = datum["sequence_key"]
            h5py_grid_map = self.h5py_handles[sk][sk][gridmap_key]
            gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

            target_idxs = torch.tensor(
                [
                    gm_layers.index(l.name)
                    for l in self.cfg.target_layers.values()
                    if l.gridmap_topic == gridmap_key
                ]
            )
            aux_idxs = torch.tensor(
                [
                    gm_layers.index(l.name)
                    for l in self.cfg.aux_layers.values()
                    if l.gridmap_topic == gridmap_key
                ]
            )

            target_out_idxs = torch.tensor(
                [
                    j
                    for j, l in enumerate(self.cfg.target_layers.values())
                    if l.gridmap_topic == gridmap_key
                ]
            )
            aux_out_idxs = torch.tensor(
                [
                    j
                    for j, l in enumerate(self.cfg.aux_layers.values())
                    if l.gridmap_topic == gridmap_key
                ]
            )
            grid_map_resolution = torch.tensor(h5py_grid_map["resolution"][0])
            np_data = np.array(h5py_grid_map[f"data"][gm_idx])  # [gm_idx]{gm_idx}
            H_sensor_gravity__map = torch.from_numpy(
                np.array(h5py_grid_map[f"T_sensor_gravity__map"][gm_idx])
            )

            grid_map_data = torch.from_numpy(
                np.ascontiguousarray(np.ascontiguousarray(np_data))
            ).float()

            if len(target_idxs) != 0:
                target[target_out_idxs] = grid_map_data[target_idxs]
            if len(aux_out_idxs) != 0:
                aux[aux_out_idxs] = grid_map_data[aux_idxs]

        target *= self.target_scale[:, None, None]
        target = target.clip(
            self.target_clip_min[:, None, None], self.target_clip_max[:, None, None]
        )

        aux *= self.aux_scale[:, None, None]
        aux = aux.clip(
            self.aux_clip_min[:, None, None], self.aux_clip_max[:, None, None]
        )

        # raw_ele_idx = 1
        # aux = self.replace_nans_with_nearest_neighbor(aux, raw_ele_idx)

        return target, aux, H_sensor_gravity__map, grid_map_resolution

    def replace_nans_with_nearest_neighbor(self, data_original, idx):
        data = data_original[idx]
        data = np.array(data)
        h, w = data.shape[:2]
        mask = np.isnan(data)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        known_x = xx[~mask]
        known_y = yy[~mask]
        known_v = data[~mask]
        missing_x = xx[mask]
        missing_y = yy[mask]
        interp_values = interpolate.griddata(
            (known_x, known_y),
            known_v,
            (missing_x, missing_y),
            method="nearest",
            fill_value=0,
        )
        interp_image = data.copy()
        interp_image[missing_y, missing_x] = interp_values
        data_original[idx] = torch.from_numpy(interp_image)

        return data_original

    def get_sequence_key(self, index, tag_with_front_camera):
        # Is called outside from dataloader to peform stamping in visualization
        index_new = self.index_mapping[index]
        res = self.dataset_config[index_new]["sequence_key"]
        if tag_with_front_camera:
            res += "_" + str(self.dataset_config[index_new]["image_front"]).zfill(6)
        return res

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

        if self.mode == "train" and self.cfg.augment_gridmap:
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
        }


def collate_fn(batch):
    """
    Returns a batch of data from the dataloader
    The batch output is a dictionary with keys: imgs, rots, trans, etc (See __getitem__)
    The values would be Tensors of shape [B, ...] or dictionary for pointcloud/gvom data
    """
    output_batch = {}

    for key, values in batch[0].items():
        if type(values) != dict:
            output_batch[key] = torch.stack([item[key] for item in batch])
        else:
            # dicts are raw pointclouds
            res = {}
            stacked_scans_ls = []
            stacked_scan_indexes = []

            for j in range(len(batch)):
                stacked_scans_ls.append(torch.cat(batch[j][key]["points"]))
                stacked_scan_indexes.append(
                    torch.tensor([scan.shape[0] for scan in batch[j][key]["points"]])
                )

            res["points"] = torch.cat(stacked_scans_ls)
            res["scan"] = torch.cat(stacked_scan_indexes)
            res["batch"] = torch.stack(stacked_scan_indexes).sum(1)

            output_batch[key] = res

    return output_batch
