import sys
import argparse
import rospy
import numpy as np
from tqdm import tqdm
import cv2
from os.path import join, splitext

import torch
from torchvision.transforms.functional import affine
from tqdm import tqdm
import h5py
import copy
import signal
import sys
from torchvision.transforms.functional import pad

from utils.h5py_writer import DatasetWriter
from utils.loading import load_pkl
from perception_bev_learning.ros import SimpleNumpyToRviz

from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pandas as pd
import pickle
from scripts.preprocessing_new.utils.tf_utils import get_np_T
from PIL import Image
from matplotlib import cm
import matplotlib
import imageio


class GenerateDepthSupevision:
    def __init__(self, cfg):
        self.cfg = OmegaConf.load(cfg)
        OmegaConf.resolve(self.cfg)
        self.cfg = instantiate(self.cfg)
        self.write = self.cfg.write
        self.min_dist = self.cfg.min_dist
        # Dictionary for all image keys to depth keys
        self.image_keys = self.cfg.image_keys
        self.depth_keys = self.cfg.depth_keys
        self.image_depth_dict = {
            self.image_keys[i]: self.depth_keys[i] for i in range(len(self.image_keys))
        }
        self.pointcloud_key = self.cfg.pointcloud_key
        self.mask_path = self.cfg.mask_path
        # create a dictionary of the masks

        self.image_mask_dict = {
            self.image_keys[i]: imageio.imread(
                self.mask_path + "/" + self.image_keys[i] + "_mask.png"
            )
            for i in range(len(self.image_keys))
        }

        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self):
        try:
            self.file.close()
        except:
            pass
        sys.exit()

    def compute(self, h5py_file, pkl_file):
        self.file = h5py.File(h5py_file, "r+")
        self.pd_df = pd.DataFrame(load_pkl(pkl_file))

        # Create gt_key colums if they don't exist
        headers = self.pd_df.columns.tolist()
        for depth_key in self.depth_keys:
            if depth_key not in headers:
                self.pd_df[depth_key] = np.zeros(self.pd_df.shape[0], dtype=int)

        if len(self.file.keys()) > 1:
            # Multiple can be processed but for now it is expected that H5py are separate for each seq
            print("Multiple sequences present in h5py file. Not processing further")
            exit()

        seq_name = list(self.file.keys())[0]
        h5py_seq = self.file[seq_name]
        h5py_pointcloud = h5py_seq[self.pointcloud_key]

        # Extract pickle config file for h5py seq
        df = self.pd_df[self.pd_df["sequence_key"] == seq_name]
        df_len = df.shape[0]
        # idx_mapping = np.arange(0, df_len)
        idx_mapping = df.index.tolist()

        # Set the correct config indices
        for depth_key in self.depth_keys:
            self.pd_df.loc[df.index, depth_key] = np.arange(0, df_len)

        if self.write:
            dataset_writer = DatasetWriter(h5py_file, open_file=self.file)

        try:
            with tqdm(
                total=len(idx_mapping),
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
            ) as pbar:
                for idx in idx_mapping:
                    datum = self.pd_df.iloc[idx]

                    for img_key in self.image_keys:
                        camera_info_key = str(img_key) + "-camera_info"
                        h5py_image = h5py_seq[img_key]
                        h5py_camera_info = h5py_seq[camera_info_key]

                        image_idx = datum[img_key]
                        img_arr = np.array(h5py_image[f"image"][image_idx])

                        K = np.array(h5py_camera_info["P"]).reshape(3, 4)[:3, :3]
                        # K = np.array(h5py_camera_info["K"]).reshape(3,3)
                        # TODO something went wrong with some Camp Roberts DS so correcting here
                        if K[0, 2] == 640:
                            K[:2, :] *= 0.5

                        H_map__camera = get_np_T(
                            t=h5py_image[f"tf_translation"][image_idx],
                            q=h5py_image[f"tf_rotation_xyzw"][image_idx],
                        )

                        points_baselink = []
                        H_array = []
                        for idx_pointcloud in datum[self.pointcloud_key]:
                            H_map__base_link = get_np_T(
                                t=h5py_pointcloud[f"tf_translation"][idx_pointcloud],
                                q=h5py_pointcloud[f"tf_rotation_xyzw"][idx_pointcloud],
                            )

                            valid_point = np.array(
                                h5py_pointcloud[f"valid"][idx_pointcloud]
                            ).sum()
                            x = h5py_pointcloud[f"x"][idx_pointcloud][:valid_point]
                            y = h5py_pointcloud[f"y"][idx_pointcloud][:valid_point]
                            z = h5py_pointcloud[f"z"][idx_pointcloud][:valid_point]
                            points = np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)

                            H_map__base_link = np.tile(
                                H_map__base_link[None, :, :], (valid_point, 1, 1)
                            )

                            points_baselink.append(points)
                            H_array.append(H_map__base_link)

                        points_baselink = np.vstack(points_baselink)
                        H_array = np.vstack(H_array)

                        points_map = np.matmul(H_array, points_baselink[:, :, None])[
                            :, :, 0
                        ]
                        # Now we transform them to the current camera frame
                        points_camera = (np.linalg.inv(H_map__camera) @ points_map.T).T
                        # Now we project the points using the K Matrix
                        points_image = (K @ points_camera[:, :3].T).T
                        # Now we save the depths and normalize the pixel co-ordinates
                        depths = points_image[:, 2]
                        points_xy = points_image[:, :2] / points_image[:, 2:3]

                        # Apply the maskings
                        # Remove points that are either outside or behind the camera.
                        # Leave a margin of 1 pixel for aesthetic reasons. Also make
                        # sure points are at least 1m in front of the camera to avoid
                        # seeing the lidar points on the camera casing for non-keyframes
                        # which are slightly out of sync.
                        mask_veh = self.image_mask_dict[img_key] > 0
                        mask_veh = np.all(mask_veh, axis=-1)

                        mask = np.ones(depths.shape[0], dtype=bool)
                        mask = np.logical_and(mask, depths > self.cfg.min_dist)
                        mask = np.logical_and(
                            mask, points_xy[:, 0] < img_arr.shape[1] - 1
                        )
                        mask = np.logical_and(mask, points_xy[:, 0] > 1)
                        mask = np.logical_and(
                            mask, points_xy[:, 1] < img_arr.shape[0] - 1
                        )
                        mask = np.logical_and(mask, points_xy[:, 1] > 1)
                        points_xy = points_xy[mask, :]
                        depths = depths[mask]

                        sorted_indices = np.argsort(depths)[::-1]
                        points_xy = points_xy[sorted_indices]
                        depths = depths[sorted_indices]

                        normalized_depth = np.log(depths) / 5

                        points_2d = np.round(points_xy).astype(int)

                        # Create an image with zeros (black background)
                        depth_image = np.zeros(
                            (img_arr.shape[0], img_arr.shape[1], 1), dtype=np.float32
                        )
                        depth_image[points_2d[:, 1], points_2d[:, 0]] = depths[:, None]
                        depth_image[~mask_veh] = 0

                        # # Choose a colormap
                        # cmap = cm.get_cmap('viridis')  # You can choose other colormaps
                        # colors = (cmap(normalized_depth)* 255).astype(np.uint8)
                        # colors = colors[:, :3]
                        # colors = colors[:,::-1]
                        # color_image = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8)
                        # color_image[points_2d[:, 1], points_2d[:, 0]] = colors
                        # color_image[~mask_veh] = 0

                        depth_key = self.image_depth_dict[img_key]
                        # overlay = cv2.addWeighted(img_arr, 0.5, color_image, 0.5, 0)
                        # overlay = overlay[:,:,::-1]
                        # im = Image.fromarray(overlay)
                        # im.save(f'images/depth_image_{idx}_{depth_key}.png')

                        # Write to H5
                        if self.write:
                            # pass
                            dynamic = {"data": depth_image}
                            dataset_writer.add_data(
                                sequence=seq_name,
                                fieldname=depth_key,
                                data_dict=dynamic,
                            )

                    pbar.update(1)

        except Exception as e:
            self.file.close()
            print(e)
            raise Exception(e)

        self.file.close()

    def __del__(self):
        self.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add GT Fused Gridmaps")
    parser.add_argument("--cfg", type=str, help="path to config yaml file ")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # Get the h5py files
    if Path(cfg.h5_files).is_dir():
        h5py_files = [str(s) for s in Path(cfg.h5_files).rglob("*.h5py")]
        h5py_files.sort(reverse=True)
    elif splitext(cfg.h5_files)[-1] == ".h5py":
        h5py_files = [cfg.h5_files]
    else:
        print("Correct path for h5py files is not provided in cfg")
        exit()

    print(f"The following h5py files will be processed: {h5py_files}")

    for f in h5py_files:
        print("Processing file: ", f)
        pkl_cfg_file = splitext(f)[0] + ".pkl"
        print("Config file is: ", pkl_cfg_file)
        gt_sup = GenerateDepthSupevision(cfg=args.cfg)
        gt_sup.compute(h5py_file=f, pkl_file=pkl_cfg_file)

        # Save the Pandas DF
        list_of_dicts = gt_sup.pd_df.to_dict(orient="records")
        with open(pkl_cfg_file, "wb") as handle:
            pickle.dump(list_of_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished processing all files")
