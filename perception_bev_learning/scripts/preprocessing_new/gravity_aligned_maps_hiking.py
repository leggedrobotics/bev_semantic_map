import sys
import argparse
import numpy as np
from tqdm import tqdm
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
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pandas as pd
import pickle

from icp_register import (
    preprocess_point_cloud,
    execute_icp,
    execute_global_registration,
    draw_registration_result,
)

import open3d as o3d

from torch.nn import functional as F
from scipy import ndimage

from utils.tf_utils import get_yaw_oriented_sg, connected_component_search, get_H_h5py, inv

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


class GravityAlignedGridMaps:
    def __init__(self, cfg):
        self.cfg = OmegaConf.load(cfg)
        OmegaConf.resolve(self.cfg)
        self.cfg = instantiate(self.cfg)

        self.write = self.cfg.write
        self.gridmap_size = self.cfg.HW_ext
        self.icp_cfg = self.cfg.icp_config

        # Dictionary for all gridmap objects to be converted
        self.input_gridmap_keys = self.cfg.input_layer_keys
        self.ouput_gridmap_keys = self.cfg.output_layer_keys

        assert len(self.input_gridmap_keys) == len(self.ouput_gridmap_keys)

        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self):
        try:
            self.file.close()
        except:
            pass
        sys.exit()
    
    def get_pointcloud_data(self, h5py_pointcloud, idx, H_sensor_gravity__map):

        try:
            idx_pointcloud = idx[-1]
        except:
            idx_pointcloud = idx

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

        return points_sensor_origin

    def compute(self, h5py_file, pkl_file):
        self.file = h5py.File(h5py_file, "r+")
        self.pd_df = pd.DataFrame(load_pkl(pkl_file))

        # Create gt_key colums if they don't exist
        headers = self.pd_df.columns.tolist()
        for out_key, in_key in zip(self.ouput_gridmap_keys, self.input_gridmap_keys):
            if out_key not in headers:
                self.pd_df[out_key] = self.pd_df[self.cfg.anchor_key]
        
        if "fitness" not in headers:
            self.pd_df["fitness"] = np.zeros_like(self.pd_df[self.cfg.anchor_key])
        
        if "is_valid" not in headers:
            self.pd_df["is_valid"] = np.full(self.pd_df.shape[0], False, dtype=bool)

        if len(self.file.keys()) > 1:
            # Multiple can be processed but for now it is expected that H5py are separate for each seq
            print("Multiple sequences present in h5py file. Not processing further")
            exit()

        seq_name = list(self.file.keys())[0]

        # Extract pickle config file for h5py seq
        # df = self.pd_df[self.pd_df["sequence_key"] == seq_name]

        mask = self.pd_df["sequence_key"] == seq_name
        df = self.pd_df[mask]
        df_len = df.shape[0]

        idx_mapping = df.index.tolist()

        # df_len = df.shape[0]
        # idx_mapping = np.arange(0, df_len)
        # # idx_mapping = df.index.tolist()

        h5py_seq = self.file[seq_name]
        if self.write:
            dataset_writer = DatasetWriter(h5py_file, open_file=self.file)

        for out_key, in_key in zip(self.ouput_gridmap_keys, self.input_gridmap_keys):
            # del h5py_seq[out_key]

            print(f"Input key is {in_key}")
            print(f"Present keys are {h5py_seq.keys()}")
            h5py_gridmap = h5py_seq[in_key]
            h5py_anchor = h5py_seq[self.cfg.anchor_key]
            h5py_pcl = h5py_seq[self.cfg.pointcloud_key]
            gm_layers = [g.decode("utf-8") for g in h5py_gridmap["layers"]]

            length_meters = np.array(h5py_gridmap[f"length"])
            resolution = h5py_gridmap["resolution"][0]

            N, C, H, W = h5py_gridmap["data"].shape
            H_ext, W_ext = self.gridmap_size
            pad_val = [int((H_ext - H) / 2), int((W_ext - W) / 2)]

            elevation_idxs = np.array(
                [
                    gm_layers.index(l_name)
                    for l_name in self.cfg.elevation_layers
                    if l_name in gm_layers
                ]
            )

            try:
                with tqdm(
                    total=len(idx_mapping),
                    desc="Total",
                    colour="green",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                ) as pbar:
                    for idx in idx_mapping:
                        anchor_idx = self.pd_df.at[idx, self.cfg.anchor_key]
                        gm_idx = self.pd_df.at[idx, in_key]
                        pcl_idx = self.pd_df.at[idx, self.cfg.pointcloud_key]

                        H_map__sensor_origin_link = get_H_h5py(
                            t=h5py_anchor[f"tf_translation_map_o3d_localization_manager__base"][anchor_idx],
                            q=h5py_anchor[f"tf_rotation_xyzw_map_o3d_localization_manager__base"][anchor_idx],
                        )
                        
                        H_map__footprint = get_H_h5py(
                            t=h5py_anchor[f"tf_translation_map_o3d_localization_manager__footprint"][anchor_idx],
                            q=h5py_anchor[f"tf_rotation_xyzw_map_o3d_localization_manager__footprint"][anchor_idx],
                        )

                        H_map__sensor_gravity = torch.eye(4)
                        H_map__sensor_gravity[:3,:3] = H_map__footprint[:3,:3]
                        H_map__sensor_gravity[:3,3] = H_map__sensor_origin_link[:3, 3]
                        H_map_sgyaw = get_yaw_oriented_sg(H_map__sensor_gravity)
                        H_sensor_gravity__map = inv(H_map__sensor_gravity)
                        H_sgyaw__map = inv(H_map_sgyaw)

                        # Assuming that the X and Y lengths are the same
                        length_x = length_meters[0] // resolution
                        length_y = length_meters[1] // resolution

                        origin = np.array(h5py_gridmap[f"position"][gm_idx])
                        np_data = np.array(h5py_gridmap[f"data"][gm_idx]) 

                        # Assuming that there is only 1 elevation layer
                        elevation_data = np_data[elevation_idxs[0]]

                        # Filter out the unconnected components in the map (Which are mostly outliers from overhanging objects)
                        connected_elevation_map = connected_component_search(elevation_data, threshold=self.cfg.connected_component_threshold)
                        
                        # Extract the traversability layer which we will use to filter out the gridmap points used for ICP
                        # Since we do not want to use all gridmap points for the ICP (Only the ones on the trail are more relevant)
                        trav_idx = gm_layers.index(self.icp_cfg.trav_layer)
                        nav_trav_map = np_data[trav_idx]

                        additional_fields_data = np.delete(np_data, elevation_idxs[0], axis=0)
                        
                        # Prepare the point cloud from the grid map
                        indices = np.where(~np.isnan(connected_elevation_map))

                        points = np.column_stack(
                        (
                            resolution * indices[0] - resolution * length_x // 2,
                            resolution * indices[1] - resolution * length_y // 2,
                            connected_elevation_map[indices],
                        )
                        )

                        x_points = points[:, 0] + origin[0]
                        y_points = points[:, 1] + origin[1]
                        z_points = points[:, 2]

                        # Filter out the points which satisfy the traversability threshold
                        points_trav = nav_trav_map[indices]
                        trav_indices = points_trav > self.icp_cfg.trav_threshold

                        points_map = np.vstack((x_points, y_points, z_points, np.ones_like(x_points)))
                        points_sg = (np.array(H_sgyaw__map) @ points_map).T

                        ############################################

                        # For ICP
                        pcd_data = self.get_pointcloud_data(h5py_pcl, pcl_idx, H_sgyaw__map)

                        pcd_data_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_data))
                        
                        pointsg_data_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sg[:,:3]))
                        
                        # Trav Filtered gridmap points to be used for ICP 
                        pointsg_data_o3d_icp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sg[trav_indices][:,:3]))

                        # pcd_data_o3d = o3d.geometry.PointCloud()
                        # pointsg_data_o3d = o3d.geometry.PointCloud()
                        # pointsg_data_o3d_icp = o3d.geometry.PointCloud()
                        # # Convert the numpy array to an Open3D Vector3dVector
                        # pcd_data_o3d.points = o3d.utility.Vector3dVector(pcd_data)
                        # pointsg_data_o3d.points = o3d.utility.Vector3dVector(points_sg[:,:3])
                        # pointsg_data_o3d_icp.points = o3d.utility.Vector3dVector(points_sg[trav_indices][:,:3])
                    
                        source_down = preprocess_point_cloud(pointsg_data_o3d_icp, self.icp_cfg.voxel_size_gridmap)
                        target_down = preprocess_point_cloud(pcd_data_o3d, self.icp_cfg.voxel_size_pcl)
                        result_ransac = execute_icp(
                            source_down, target_down, self.icp_cfg.voxel_size_icp
                        )

                        print(f"RMSE: {result_ransac.inlier_rmse}")
                        print(f"Fitness: {result_ransac.fitness}")
                        result_tf = np.array(result_ransac.transformation)
                        result_ypr = R.from_matrix(result_tf[:3, :3]).as_euler(
                            "xyz", degrees=True
                        )
                        print(f"RPY (degrees) is {result_ypr} \n Translation is {result_tf[:3,3]}")
                        # draw_registration_result(source_down, target_down, result_ransac.transformation)

                        # Transform the gridmap points according to the result
                        pointsg_data_o3d = pointsg_data_o3d.transform(
                            result_ransac.transformation
                        )
                        points_sg = np.asarray(pointsg_data_o3d.points)

                        # Save the Fitness value to the df
                        self.pd_df.at[idx, "fitness"] = result_ransac.fitness
                        self.pd_df.at[idx, "is_valid"] = result_ransac.fitness > self.icp_cfg.fitness

                        ############################################

                        new_shape = (int(length_y * 1.42), int(length_x * 1.42))
                        new_grid_map = np.full(new_shape, np.nan)

                        additional_fields_values = additional_fields_data[:, indices[0], indices[1]]
                        new_additional_fields = [np.full(new_shape, np.nan) for _ in range(len(additional_fields_data))]

                        values = points_sg[:, 2]  # Assuming z-values represent the values in the gridmap

                        # Normalize coordinates to match the grid indices
                        normalized_coords = (
                            (points_sg[:, :2] / resolution) + np.array(new_shape) / 2
                        ).astype(int)

                        # Remove indices that are out of bounds
                        valid_indices = np.logical_and(
                            np.logical_and(normalized_coords[:, 0] >= 0, normalized_coords[:, 0] < new_shape[0]),
                            np.logical_and(normalized_coords[:, 1] >= 0, normalized_coords[:, 1] < new_shape[1])
                        )
                        normalized_coords = normalized_coords[valid_indices]
                        values = values[valid_indices]
                        additional_fields_values = additional_fields_values[:, valid_indices]

                        new_grid_map[normalized_coords[:, 0], normalized_coords[:, 1]] = values

                        for layer_idx in range(len(new_additional_fields)):
                            new_additional_fields[layer_idx][normalized_coords[:, 0], normalized_coords[:, 1]] = additional_fields_values[layer_idx]

                        final_gridmap = np.insert(new_additional_fields, elevation_idxs[0], new_grid_map[...,], axis=0)


                        ################################################
                        ## Nearest Neighbor interpolation for missing values within a pixel radius

                        # Step 1: Create a mask of NaNs
                        nan_mask = np.isnan(final_gridmap)

                        # Step 2: Use distance transform to find the nearest non-NaN value
                        # Distances and indices of the nearest non-NaN values
                        distances, nearest_indices = ndimage.distance_transform_edt(nan_mask, return_indices=True)
                        
                        # Get the nearest values for NaN positions
                        nearest_values = final_gridmap[tuple(nearest_indices)]
                        
                        # Step 3: Ensure replacements are only made within the specified max_distance
                        within_distance_mask = distances <= 1
                        
                        # Step 4: Combine the masks and fill NaNs
                        fill_mask = nan_mask & within_distance_mask
                        final_gridmap[fill_mask] = nearest_values[fill_mask]

                        final_gridmap = center_crop(torch.Tensor(final_gridmap)[None], self.gridmap_size)

                        ############################

                        static = {
                            "layers": gm_layers,
                            "resolution": np.array(h5py_gridmap["resolution"]),
                            "length": np.array((resolution * H_ext, resolution * W_ext)).reshape(-1),
                            "header_frame_id": "sensor_gravity_yaw",
                        }
                        dynamic = {
                            "data": np.array(final_gridmap),
                            "header_seq": h5py_gridmap["header_seq"][gm_idx],
                            "header_stamp_nsecs": h5py_gridmap["header_stamp_nsecs"][
                                gm_idx
                            ],
                            "header_stamp_secs": h5py_gridmap["header_stamp_secs"][
                                gm_idx
                            ],
                            "orientation_xyzw": h5py_gridmap["orientation_xyzw"][
                                gm_idx
                            ],
                            "position": h5py_gridmap["position"][gm_idx],
                            "tf_rotation_xyzw": h5py_gridmap["tf_rotation_xyzw"][
                                gm_idx
                            ],
                            "tf_translation": h5py_gridmap["tf_translation"][gm_idx],
                            "T_sensor_gravity_yaw__map": np.array(H_sgyaw__map),
                        }
                        if self.write:
                            dataset_writer.add_static(
                                sequence=seq_name,
                                fieldname=out_key,
                                data_dict=static,
                            )
                            dataset_writer.add_data(
                                sequence=seq_name,
                                fieldname=out_key,
                                data_dict=dynamic,
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
        description="Convert Gridmaps to Gravity aligned frame and fuse Digital Elevation Maps"
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
        gt_sup = GravityAlignedGridMaps(cfg=args.cfg)
        gt_sup.compute(h5py_file=f, pkl_file=pkl_cfg_file)

        # Save the Pandas DF
        list_of_dicts = gt_sup.pd_df.to_dict(orient="records")
        with open(pkl_cfg_file, "wb") as handle:
            pickle.dump(list_of_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished processing all files")
