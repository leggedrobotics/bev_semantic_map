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
from perception_bev_learning.utils import get_H_h5py, inv, get_gravity_aligned
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pandas as pd
import pickle

from icp_register import (
    preprocess_point_cloud,
    execute_icp,
    gridmap_to_pointcloud,
    pointcloud_to_gridmap,
    execute_global_registration,
    draw_registration_result,
)
from shapely.geometry import Polygon
from shapely.affinity import rotate
from pyproj import Proj, transform, Transformer
import rasterio
import rasterio.mask
from torch.nn import functional as F
from scipy.ndimage import binary_dilation, gaussian_filter


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

        self.visu = self.cfg.visualize
        self.write = self.cfg.write
        self.gridmap_size = self.cfg.HW_ext
        self.dem_cfg = self.cfg.dem_config

        if self.visu:
            self.vis_pred = SimpleNumpyToRviz()
            self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")
            rospy.on_shutdown(self.shutdown_callback)

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

    def compute(self, h5py_file, pkl_file):
        self.file = h5py.File(h5py_file, "r+")
        self.pd_df = pd.DataFrame(load_pkl(pkl_file))

        # Create gt_key colums if they don't exist
        headers = self.pd_df.columns.tolist()
        for out_key, in_key in zip(self.ouput_gridmap_keys, self.input_gridmap_keys):
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

        for out_key, in_key in zip(self.ouput_gridmap_keys, self.input_gridmap_keys):
            # del h5py_seq[out_key]

            print(f"Input key is {in_key}")
            print(f"Present keys are {h5py_seq.keys()}")
            h5py_gridmap = h5py_seq[in_key]
            h5py_anchor = h5py_seq[self.cfg.anchor_key]
            layers = [l.decode("utf-8") for l in h5py_gridmap["layers"]]

            if in_key == self.cfg.dem_layer_key:
                # We need to do DEM processing and fusion
                # Add the new layers name in the layers list
                layers.append(self.cfg.dem_layer_name)
                self.dem = rasterio.open(self.dem_cfg.input_dem)
                self.utmProj = Proj(
                    proj="utm", zone=10, preserve_units=False, ellps="WGS84"
                )
                self.transformer = Transformer.from_crs(
                    "EPSG:4326", self.dem.crs, always_xy=True
                )
                h5py_o_gps = h5py_seq[self.cfg.gps_o_key]
                h5py_p_gps = h5py_seq[self.cfg.gps_p_key]
                ele_idx_dem = layers.index(self.dem_cfg.elevation_layer)
                # ele_est_idx_dem = layers.index(self.dem_cfg.elevation_layer_est)
                conf_idx_dem = layers.index(self.dem_cfg.confidence_layer)

            length = np.array(h5py_gridmap["length"])
            res = np.array(h5py_gridmap["resolution"])

            N, C, H, W = h5py_gridmap["data"].shape
            H_ext, W_ext = self.gridmap_size
            pad_val = [int((H_ext - H) / 2), int((W_ext - W) / 2)]

            elevation_idxs = torch.tensor(
                [
                    layers.index(l_name)
                    for l_name in self.cfg.elevation_layers
                    if l_name in layers
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
                        anchor_idx = df.at[idx, self.cfg.anchor_key]
                        gm_idx = df.at[idx, in_key]

                        H_map__sensor_origin_link = get_H_h5py(
                            t=h5py_anchor[f"tf_translation_map__sensor_origin_link"][
                                anchor_idx
                            ],
                            q=h5py_anchor[f"tf_rotation_xyzw_map__sensor_origin_link"][
                                anchor_idx
                            ],
                        )
                        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
                        H_sensor_gravity__map = get_gravity_aligned(
                            H_sensor_origin_link__map
                        )

                        H_map__grid_map_center = torch.eye(4)
                        H_map__grid_map_center[:3, 3] = torch.tensor(
                            h5py_gridmap[f"position"][gm_idx]
                        )

                        grid_map_resolution = torch.tensor(
                            h5py_gridmap["resolution"][0]
                        )
                        H_sensor_gravity__grid_map_center = (
                            H_sensor_gravity__map @ H_map__grid_map_center
                        )

                        yaw = R.from_matrix(
                            H_sensor_gravity__grid_map_center.clone().numpy()[:3, :3]
                        ).as_euler(seq="zyx", degrees=True)[0]
                        shift = (
                            H_sensor_gravity__grid_map_center[:2, 3]
                        ) / grid_map_resolution
                        sh = [shift[1], shift[0]]

                        np_data = np.array(h5py_gridmap[f"data"][gm_idx])
                        H_c, W_c = int(np_data.shape[1] / 2), int(np_data.shape[2] / 2)

                        grid_map_data = torch.from_numpy(
                            np.ascontiguousarray(np.ascontiguousarray(np_data))
                        )

                        grid_map_data[elevation_idxs] = (
                            grid_map_data[elevation_idxs]
                            + H_sensor_gravity__grid_map_center[2, 3]
                        )

                        grid_map_data_rotated = affine(
                            grid_map_data[None],
                            angle=-yaw,
                            translate=sh,
                            scale=1,
                            shear=0,
                            center=(H_c, W_c),
                            fill=torch.nan,
                        )[0]

                        # If pad-value is positive, apply padding otherwise apply center crop
                        if pad_val[0] > 0:
                            grid_map_data_rotated = pad(
                                grid_map_data_rotated, pad_val, torch.nan, "constant"
                            )
                        else:
                            grid_map_data_rotated = center_crop(
                                grid_map_data_rotated, self.gridmap_size
                            )

                        grid_map_data_rotated = np.array(grid_map_data_rotated)
                        # For DEM Fusion #########################
                        if in_key == self.cfg.dem_layer_key:
                            # try:
                            gps_o_idx = df.at[idx, self.cfg.gps_o_key]
                            gps_p_idx = df.at[idx, self.cfg.gps_p_key]
                            lat_gps = h5py_p_gps["latitude"][gps_o_idx]
                            long_gps = h5py_p_gps["longitude"][gps_o_idx]
                            ele_gps = h5py_p_gps["altitude"][gps_o_idx]
                            orientation = h5py_o_gps["rotation_xyzw"][gps_p_idx]
                            rotation_matrix = R.from_quat(orientation).as_matrix()
                            euler_angles = R.from_matrix(rotation_matrix).as_euler(
                                "xyz", degrees=True
                            )
                            heading_degrees = euler_angles[2]
                            xx, yy = self.transformer.transform(long_gps, lat_gps)
                            row, col = self.dem.index(xx, yy)
                            wg = self.dem_cfg.width_gridmap / 2.0
                            crop_polygon = Polygon(
                                (
                                    (xx - wg, yy + wg),
                                    (xx + wg, yy + wg),
                                    (xx + wg, yy - wg),
                                    (xx - wg, yy - wg),
                                )
                            )
                            crop_polygon_rot = rotate(
                                crop_polygon, -1 * heading_degrees, (xx, yy)
                            )
                            out_image, out_transform = rasterio.mask.mask(
                                self.dem, [crop_polygon_rot], crop=True
                            )
                            out_image[out_image < -100] = np.max(out_image)
                            grid_map_data_sat = torch.from_numpy((out_image)).squeeze(0)
                            grid_map_data_sat = grid_map_data_sat - ele_gps
                            grid_map_data_sat_rotated = affine(
                                grid_map_data_sat[None],
                                angle=-heading_degrees,
                                translate=[0, 0],
                                scale=1,
                                shear=0,
                                center=(
                                    grid_map_data_sat.shape[0] // 2,
                                    grid_map_data_sat.shape[1] // 2,
                                ),
                                fill=torch.nan,
                            )[0]

                            grid_map_data_sat_rotated = center_crop(
                                grid_map_data_sat_rotated, self.dem_cfg.crop_gridmap
                            )
                            elevation_values = np.array(grid_map_data_sat_rotated[:, :])
                            elevation_gt = grid_map_data_rotated[ele_idx_dem]
                            # elevation_est_gt = grid_map_data_rotated[ele_est_idx_dem]
                            confidence_gt = grid_map_data_rotated[conf_idx_dem]
                            # ele_veh_gt = elevation_gt[elevation_gt.shape[0]//2, elevation_gt.shape[1]//2]
                            ele_veh_gt = -1.5
                            ele_veh_sat = elevation_values[
                                elevation_values.shape[0] // 2,
                                elevation_values.shape[1] // 2,
                            ]

                            elevation_values = elevation_values - (
                                ele_veh_sat - ele_veh_gt
                            )  # Z offset for Altitude compensation
                            elevation_values = elevation_values[
                                ::-1, ::-1
                            ]  # To bring in same axis
                            gridmap1 = elevation_gt
                            gridmap2 = elevation_values
                            ######## ICP #####################
                            pointcloud1 = gridmap_to_pointcloud(gridmap1)
                            pointcloud2 = gridmap_to_pointcloud(gridmap2, multiplier=1)

                            source_down = preprocess_point_cloud(
                                pointcloud2, self.dem_cfg.voxel_size
                            )
                            target_down = preprocess_point_cloud(
                                pointcloud1, self.dem_cfg.voxel_size
                            )
                            result_ransac = execute_icp(
                                source_down, target_down, self.dem_cfg.voxel_size
                            )

                            # source_down, source_fpfh = preprocess_point_cloud(pointcloud2, self.dem_cfg.voxel_size, return_features=True)
                            # target_down, target_fpfh = preprocess_point_cloud(pointcloud1, self.dem_cfg.voxel_size, return_features=True)
                            # result_ransac = execute_global_registration(source_down, target_down,
                            #                         source_fpfh, target_fpfh,
                            #                         self.dem_cfg.voxel_size)

                            result_tf = np.array(result_ransac.transformation)
                            result_ypr = R.from_matrix(result_tf[:3, :3]).as_euler(
                                "xyz", degrees=True
                            )
                            # print(f"RPY (degrees) is {result_ypr} \n Translation is {result_tf[:3,3]}")
                            # draw_registration_result(source_down, target_down, result_ransac.transformation)
                            pointcloud2 = pointcloud2.transform(
                                result_ransac.transformation
                            )
                            gridmap2_new = pointcloud_to_gridmap(
                                pointcloud2,
                                multiplier=1,
                                grid_shape=self.dem_cfg.crop_gridmap_tolerance,
                            )
                            gridmap2_new = F.interpolate(
                                torch.from_numpy(gridmap2_new)
                                .unsqueeze(0)
                                .unsqueeze(0),
                                scale_factor=self.dem_cfg.scale,
                                mode="bilinear",
                                align_corners=True,
                            )
                            gridmap2_new = center_crop(
                                gridmap2_new, output_size=gridmap1.shape
                            )
                            gridmap2_new = np.array(gridmap2_new.squeeze(0).squeeze(0))

                            ######### Fusion ################

                            conf_idxs = confidence_gt < 0.8
                            # conf_idxs = np.isnan(elevation_gt)
                            gridmap_duplicate = elevation_gt.copy()

                            ###### Smoothing before Fusion ########
                            fusion_edge_mask = np.zeros_like(
                                gridmap_duplicate, dtype=bool
                            )
                            fusion_edge_mask[conf_idxs] = True
                            edges = (
                                cv2.Canny(
                                    fusion_edge_mask.astype(np.uint8) * 255, 30, 100
                                )
                                > 0
                            )
                            size = (30, 30)
                            shape = cv2.MORPH_RECT
                            kernel = cv2.getStructuringElement(shape, size)
                            edges = 10 * np.array(edges, dtype=np.float32)
                            dilated_fusion_edges = cv2.dilate(edges, kernel)
                            dilated_fusion_edges = dilated_fusion_edges > 0

                            gridmap_duplicate[conf_idxs] = gridmap2_new[conf_idxs]
                            kernel_size = 25
                            smoothed_img = cv2.GaussianBlur(
                                gridmap_duplicate, (kernel_size, kernel_size), 0
                            )
                            gridmap_duplicate[dilated_fusion_edges] = smoothed_img[
                                dilated_fusion_edges
                            ]

                            grid_map_data_rotated = np.concatenate(
                                (
                                    grid_map_data_rotated,
                                    np.expand_dims(gridmap_duplicate, axis=0),
                                )
                            )
                            # except Exception as e:
                            #     print(e)
                            #     print(f"latitude {lat_gps} and longitude {long_gps}")
                            #     print("skipping fusion for this sample")
                            #     grid_map_data_rotated = np.concatenate((grid_map_data_rotated,np.expand_dims(np.zeros_like(grid_map_data_rotated[0]), axis=0)))

                            ########################
                        static = {
                            "layers": layers,
                            "resolution": np.array(h5py_gridmap["resolution"]),
                            "length": np.array((res * H_ext, res * W_ext)).reshape(-1),
                            "header_frame_id": "sensor_gravity",
                        }
                        dynamic = {
                            "data": grid_map_data_rotated,
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
                            "T_sensor_gravity__map": np.array(H_sensor_gravity__map),
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

                        # if self.visu:
                        #     self.vis_gt.gridmap_arr(
                        #         out[:, shrink:-shrink, shrink:-shrink],
                        #         res=res,
                        #         layers=layers,
                        #         x=0,
                        #     )

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
