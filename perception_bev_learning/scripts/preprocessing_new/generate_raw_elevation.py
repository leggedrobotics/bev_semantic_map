import sys
import argparse
import rospy
import numpy as np
from tqdm import tqdm
import cv2
from os.path import join, splitext

import torch
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import affine, center_crop, pad
from tqdm import tqdm
import h5py
import copy
import signal
import sys
from torchvision.transforms.functional import pad

from utils.h5py_writer import DatasetWriter
from utils.loading import load_pkl
from perception_bev_learning.ros import SimpleNumpyToRviz
from utils.tf_utils import get_H_h5py, inv, get_gravity_aligned
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pandas as pd
import pickle

from torch.nn import functional as F
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d

def get_time(hdf5_view, j=None):
    if j is None:
        return (
            np.array(hdf5_view["header_stamp_secs"])
            + np.array(hdf5_view["header_stamp_nsecs"]) * 10**-9
        )
    else:
        return (
            np.array(hdf5_view["header_stamp_secs"][j])
            + np.array(hdf5_view["header_stamp_nsecs"][j]) * 10**-9
        )

def compute_elevation_layers_cupy(pcd_data, length, res, m, zgap, hclearance):
    # Determine the min and max values for x and y to create the grid
    x_min = - length / 2.0
    x_max = length / 2.0
    y_min = - length / 2.0
    y_max = length / 2.0

    # Create grid
    grid_x = int(cp.ceil((x_max - x_min) / res))
    grid_y = int(cp.ceil((y_max - y_min) / res))

    # Initialize elevation grids
    min_elevation = cp.full((grid_x, grid_y), cp.nan)
    max_elevation = cp.full((grid_x, grid_y), cp.nan)
    ceiling_elevation = cp.full((grid_x, grid_y), cp.nan)

    # Assign points to grid cells
    indices = ((pcd_data[:, :2] - [x_min, y_min]) / res).astype(int)

    for i in range(grid_x):
        for j in range(grid_y):
            cell_points = pcd_data[(indices[:, 0] == i) & (indices[:, 1] == j)]
            if len(cell_points) >= m:
                cell_points = cell_points[cp.argsort(cell_points[:, 2])]  # Sort by z-coordinate
                min_elevation[i, j] = cp.mean(cell_points[:m, 2])

                for k in range(len(cell_points) - 1):
                    z_diff = cell_points[k + 1, 2] - cell_points[k, 2]
                    if z_diff > zgap:
                        max_elevation[i, j] = cell_points[k, 2]
                        ceiling_elevation[i, j] = cell_points[k + 1, 2]
                        break
                else:
                    max_elevation[i, j] = min_elevation[i, j]
                    ceiling_elevation[i, j] = min_elevation[i, j] + hclearance

    return min_elevation, max_elevation, ceiling_elevation

def compute_elevation_layers_thread(pcd_data, length, res, m, zgap, hclearance):
    # Determine the min and max values for x and y to create the grid
    x_min = - length / 2.0
    x_max = length / 2.0
    y_min = - length / 2.0
    y_max = length / 2.0

    # Create grid
    grid_x = int(np.ceil((x_max - x_min) / res))
    grid_y = int(np.ceil((y_max - y_min) / res))

    # Initialize elevation grids
    min_elevation = np.full((grid_x, grid_y), np.nan)
    max_elevation = np.full((grid_x, grid_y), np.nan)
    ceiling_elevation = np.full((grid_x, grid_y), np.nan)

    # Assign points to grid cells
    indices = ((pcd_data[:, :2] - [x_min, y_min]) / res).astype(int)

    # Function to process each grid cell
    def process_grid_cell(i, j):
        cell_points = pcd_data[(indices[:, 0] == i) & (indices[:, 1] == j)]
        if len(cell_points) >= m:
            cell_points = cell_points[np.argsort(cell_points[:, 2])]  # Sort by z-coordinate
            min_elevation[i, j] = np.mean(cell_points[:m, 2])

            for k in range(len(cell_points) - 1):
                z_diff = cell_points[k + 1, 2] - cell_points[k, 2]
                if z_diff > zgap:
                    max_elevation[i, j] = cell_points[k, 2]
                    ceiling_elevation[i, j] = cell_points[k + 1, 2]
                    break
            else:
                max_elevation[i, j] = min_elevation[i, j]
                ceiling_elevation[i, j] = min_elevation[i, j] + hclearance

    # Use ThreadPoolExecutor to parallelize processing of grid cells
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_grid_cell, i, j) for i in range(grid_x) for j in range(grid_y)]

        # Wait for all threads to complete
        for future in futures:
            future.result()

    return min_elevation, max_elevation, ceiling_elevation

def compute_elevation_layers(pcd_data, length, res, m, zgap, hclearance):
    # Determine the min and max values for x and y to create the grid
    x_min = - length / 2.0
    x_max = length / 2.0
    y_min = - length / 2.0
    y_max = length / 2.0

    # Create grid
    grid_x = int(np.ceil((x_max - x_min) / res))
    grid_y = int(np.ceil((y_max - y_min) / res))

    # Initialize elevation grids
    min_elevation = np.full((grid_x, grid_y), np.nan)
    max_elevation = np.full((grid_x, grid_y), np.nan)
    ceiling_elevation = np.full((grid_x, grid_y), np.nan)

    # Assign points to grid cells
    indices = ((pcd_data[:, :2] - [x_min, y_min]) / res).astype(int)

    for i in range(grid_x):
        for j in range(grid_y):
            cell_points = pcd_data[(indices[:, 0] == i) & (indices[:, 1] == j)]
            if len(cell_points) >= m :
                cell_points = cell_points[np.argsort(cell_points[:, 2])]  # Sort by z-coordinate
                min_elevation[i, j] = np.mean(cell_points[:m, 2])
         
                for k in range(len(cell_points) - 1):
                    z_diff = cell_points[k + 1, 2] - cell_points[k, 2]
                    if z_diff > zgap:
                        max_elevation[i, j] = cell_points[k, 2]
                        ceiling_elevation[i, j] = cell_points[k + 1, 2]
                        break
                else:
                    max_elevation[i, j] = min_elevation[i, j]
                    ceiling_elevation[i, j] = min_elevation[i, j] + hclearance

    return min_elevation, max_elevation, ceiling_elevation

class RawElevationMaps:
    def __init__(self, cfg):
        self.cfg = OmegaConf.load(cfg)
        OmegaConf.resolve(self.cfg)
        self.cfg = instantiate(self.cfg)

        self.visu = self.cfg.visualize
        self.write = self.cfg.write
        self.gridmap_size = self.cfg.HW_ext

        if self.visu:
            self.vis_pred = SimpleNumpyToRviz()
            self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")
            rospy.on_shutdown(self.shutdown_callback)

        # Dictionary for all gridmap objects to be converted
        self.ouput_gridmap_layers = self.cfg.output_layer_keys
        self.output_ele_key = self.cfg.output_ele_key


        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self):
        try:
            self.file.close()
        except:
            pass
        sys.exit()

    def get_pointcloud_data(self, h5py_pointcloud, idx_pointcloud, H_sensor_gravity__map):

        H_map__base_link = get_H_h5py(
            t=h5py_pointcloud[f"tf_translation"][idx_pointcloud],  # {idx_pointcloud}
            q=h5py_pointcloud[f"tf_rotation_xyzw"][idx_pointcloud],
        )

        valid_point = np.array(h5py_pointcloud[f"valid"][idx_pointcloud]).sum()
        x = h5py_pointcloud[f"x"][idx_pointcloud][:valid_point]
        y = h5py_pointcloud[f"y"][idx_pointcloud][:valid_point]
        z = h5py_pointcloud[f"z"][idx_pointcloud][:valid_point]
        points = np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)

        H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
        points_sensor_origin = (H_sensor_gravity__base_link.numpy() @ points.T).T
        points_sensor_origin = points_sensor_origin[:, :3]

        ts = (
            h5py_pointcloud[f"header_stamp_secs"][idx_pointcloud]
            + h5py_pointcloud["header_stamp_nsecs"][idx_pointcloud] * 10**-9
        )

        return points_sensor_origin, ts


    def compute(self, h5py_file, pkl_file):
        self.file = h5py.File(h5py_file, "r+")
        self.pd_df = pd.DataFrame(load_pkl(pkl_file))

        # Create gt_key colums if they don't exist
        headers = self.pd_df.columns.tolist()

        out_key = self.output_ele_key
        if out_key not in headers:
            self.pd_df[out_key] = self.pd_df[self.cfg.anchor_key]

        if len(self.file.keys()) > 1:
            # Multiple can be processed but for now it is expected that H5py are separate for each seq
            print("Multiple sequences present in h5py file. Not processing further")
            exit()

        seq_name = list(self.file.keys())[0]

        # Extract pickle config file for h5py seq
        df = self.pd_df[self.pd_df["sequence_key"] == seq_name]
        df_len = df.shape[0]
        idx_mapping = np.arange(0, df_len)
        # idx_mapping = df.index.tolist()

        h5py_seq = self.file[seq_name]
        if self.write:
            dataset_writer = DatasetWriter(h5py_file, open_file=self.file)

        in_key = self.cfg.anchor_key
        print(f"Input key is {in_key}")
        print(f"Present keys are {h5py_seq.keys()}")

        h5py_anchor = h5py_seq[self.cfg.anchor_key]
        layers = self.cfg.output_layer_keys

        length = np.array(self.cfg.length)
        res = np.array(self.cfg.resolution)

        HW_ext = length / res

        print(HW_ext)

        try:
            with tqdm(
                total=len(idx_mapping),
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
            ) as pbar:
                for idx in idx_mapping[::10]:

                    try:
                        anchor_idx = df.at[idx, self.cfg.anchor_key][-1]
                    except:
                        anchor_idx = df.at[idx, self.cfg.anchor_key]


                    H_map__base_inverted = get_H_h5py(
                        t=h5py_anchor[f"tf_translation_map__base_inverted"][
                            anchor_idx
                        ],
                        q=h5py_anchor[f"tf_rotation_xyzw_map__base_inverted"][
                            anchor_idx
                        ],
                    )
                    
                    H_base_inverted__map = inv(H_map__base_inverted)
                    H_sensor_gravity__map = get_gravity_aligned(
                        H_base_inverted__map
                    )

                    # Now we compute the raw elevation layers
                    pcd_data , ts = self.get_pointcloud_data(h5py_anchor, anchor_idx, H_sensor_gravity__map)

                    # Now we add 10 pointclouds from the future
                    for hs_idx in range(1,10):
                        pcd , ts = self.get_pointcloud_data(h5py_anchor, anchor_idx+hs_idx, H_sensor_gravity__map)
                        pcd_data = np.concatenate((pcd_data, pcd))



                    ###### TODO: Parameters

                    m = 1  # tuning parameter to smooth out sensor noise
                    zgap = 0.5  # minimum gap between ground and ceiling
                    hclearance = 2.0  # vehicle’s vertical clearance

                    print(pcd_data.shape)

                    pcd_open3d = o3d.geometry.PointCloud()
                    pcd_open3d.points = o3d.utility.Vector3dVector(pcd_data)

                    # Define voxel size (adjust as needed)
                    voxel_size = 0.04

                    # Apply voxel grid filter
                    pcd_downsampled = pcd_open3d.voxel_down_sample(voxel_size)

                    # Convert back to numpy array if needed
                    pcd_data_np = np.asarray(pcd_downsampled.points)

                    print(pcd_data_np.shape)

                    min_elevation, max_elevation, ceiling_elevation = compute_elevation_layers_thread(pcd_data_np, length, res, m, zgap, hclearance)

                    np_data = np.stack((min_elevation, max_elevation, ceiling_elevation), axis=0)


                    # H_map__grid_map_center = torch.eye(4)
                    # H_map__grid_map_center[:3, 3] = torch.tensor(
                    #     h5py_gridmap[f"position"][gm_idx]
                    # )

                    # grid_map_resolution = torch.tensor(
                    #     h5py_gridmap["resolution"][0]
                    # )
                    # H_sensor_gravity__grid_map_center = (
                    #     H_sensor_gravity__map @ H_map__grid_map_center
                    # )

                    # yaw = R.from_matrix(
                    #     H_sensor_gravity__grid_map_center.clone().numpy()[:3, :3]
                    # ).as_euler(seq="zyx", degrees=True)[0]
                    # shift = (
                    #     H_sensor_gravity__grid_map_center[:2, 3]
                    # ) / grid_map_resolution
                    # sh = [shift[1], shift[0]]

                    # np_data = np.array(h5py_gridmap[f"data"][gm_idx])
                    # H_c, W_c = int(np_data.shape[1] / 2), int(np_data.shape[2] / 2)

                    # grid_map_data = torch.from_numpy(
                    #     np.ascontiguousarray(np.ascontiguousarray(np_data))
                    # )

                    # grid_map_data[elevation_idxs] = (
                    #     grid_map_data[elevation_idxs]
                    #     + H_sensor_gravity__grid_map_center[2, 3]
                    # )

                    # grid_map_data_rotated = affine(
                    #     grid_map_data[None],
                    #     angle=-yaw,
                    #     translate=sh,
                    #     scale=1,
                    #     shear=0,
                    #     center=(H_c, W_c),
                    #     fill=torch.nan,
                    # )[0]

                    # # If pad-value is positive, apply padding otherwise apply center crop
                    # if pad_val[0] > 0:
                    #     grid_map_data_rotated = pad(
                    #         grid_map_data_rotated, pad_val, torch.nan, "constant"
                    #     )
                    # else:
                    #     grid_map_data_rotated = center_crop(
                    #         grid_map_data_rotated, self.gridmap_size
                    #     )

                    # grid_map_data_rotated = np.array(grid_map_data_rotated)
                    
                    # ####################################

                    # static = {
                    #     "layers": layers,
                    #     "resolution": np.array(h5py_gridmap["resolution"]),
                    #     "length": np.array((res * H_ext, res * W_ext)).reshape(-1),
                    #     "header_frame_id": "sensor_gravity",
                    # }
                    # dynamic = {
                    #     "data": grid_map_data_rotated,
                    #     "header_seq": h5py_gridmap["header_seq"][gm_idx],
                    #     "header_stamp_nsecs": h5py_gridmap["header_stamp_nsecs"][
                    #         gm_idx
                    #     ],
                    #     "header_stamp_secs": h5py_gridmap["header_stamp_secs"][
                    #         gm_idx
                    #     ],
                    #     "orientation_xyzw": h5py_gridmap["orientation_xyzw"][
                    #         gm_idx
                    #     ],
                    #     "position": h5py_gridmap["position"][gm_idx],
                    #     "tf_rotation_xyzw": h5py_gridmap["tf_rotation_xyzw"][
                    #         gm_idx
                    #     ],
                    #     "tf_translation": h5py_gridmap["tf_translation"][gm_idx],
                    #     "T_sensor_gravity__map": np.array(H_sensor_gravity__map),
                    # }
                    # if self.write:
                    #     dataset_writer.add_static(
                    #         sequence=seq_name,
                    #         fieldname=out_key,
                    #         data_dict=static,
                    #     )
                    #     dataset_writer.add_data(
                    #         sequence=seq_name,
                    #         fieldname=out_key,
                    #         data_dict=dynamic,
                    #     )

                    if self.visu:
                        self.vis_gt.gridmap_arr(
                            np_data,
                            res=res,
                            layers=layers,
                            x=0,
                        )

                    pbar.update(1)

                # Delete the in_key
                # del h5py_seq[in_key]

        except Exception as e:
            self.file.close()
            print(e)
            raise Exception(e)

        self.file.close()

    def __del__(self):
        self.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VoxelMaps to raw elevation layers in gravity aligned frame"
    )
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
        raw_ele = RawElevationMaps(cfg=args.cfg)
        raw_ele.compute(h5py_file=f, pkl_file=pkl_cfg_file)

        # Save the Pandas DF
        list_of_dicts = raw_ele.pd_df.to_dict(orient="records")
        with open(pkl_cfg_file, "wb") as handle:
            pickle.dump(list_of_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished processing all files")
