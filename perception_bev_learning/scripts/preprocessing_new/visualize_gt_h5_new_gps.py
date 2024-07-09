import sys
import h5py
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm
from os.path import join
import argparse
import torch
from scipy.spatial.transform import Rotation as R
import signal
from perception_bev_learning.utils import inv, get_gravity_aligned, load_pkl, get_H_h5py
from perception_bev_learning.ros import SimpleNumpyToRviz
from pynput import keyboard
from torch.nn import functional as F
import tf
from torchvision.transforms.functional import affine, center_crop
from pyproj import Proj, transform, Transformer
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
from rasterio.windows import Window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import open3d as o3d
from icp_register import (
    preprocess_point_cloud,
    execute_global_registration,
    draw_registration_result,
    execute_icp,
    pointcloud_to_gridmap,
    gridmap_to_pointcloud,
)

matplotlib.use("TkAgg")


class GTVisualization:
    def __init__(self, h5py_path, cfg_file):
        self._bridge = CvBridge()

        dem_path = "/home/jonfrey/Downloads/camp_roberts.tif"
        # (TODO) Can potentially shift the keys to yaml file
        self.image_keys = [
            "multisense_front",
            "multisense_left",
            "multisense_right",
            "multisense_back",
        ]
        self.gridmap_key = "traversability_map_micro"
        self.anchor_key = "multisense_front"
        self.elevation_layers = ["elevation"]
        self.pointcloud_key = "velodyne_merged_points"
        self.gvom_key = "pointcloud_map-points_micro"
        self.gps_o_key = "gps_orientation"
        self.gps_p_key = "gps_pose"

        self.vis = SimpleNumpyToRviz(
            init_node=True,
            postfix="_gt",
            cv_bridge=self._bridge,
            image_keys=self.image_keys,
            node_name="GT_Vis_node",
        )

        try:
            self.dataset_config = load_pkl(cfg_file)
        except:
            print("Cant open Dataset config file")
            exit()

        try:
            self.h5py_file = h5py.File(h5py_path)
        except:
            print("Can't open h5py file")
            exit()

        self.dataset_config = [d for d in self.dataset_config]

        self.length = len(self.dataset_config)
        self.index_mapping = np.arange(0, self.length)
        # Create a transform broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.dem = rasterio.open(dem_path)
        self.utmProj = Proj(proj="utm", zone=10, preserve_units=False, ellps="WGS84")
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.dem.crs, always_xy=True
        )
        self.width_gridmap = 200  # In metres
        self.crop_gridmap = 110

    def len(self):
        return len(self.index_mapping)

    def LonLat_To_XY(self, Lon, Lat):
        """
        Inputs: Longitude, Latitude
        Returns: X, Y (Easting, Northing) for specified Zone 10
        """
        return self.utmProj(Lon, Lat)

    def XY_To_LonLat(self, x, y):
        """
        Inputs: X, Y (Easting, Northing) for specified Zone 10
        Returns: Longitude, Latitude
        """
        return self.utmProj(x, y, inverse=True)

    def get_images(self, datum):
        img_dict = {}
        ts = []
        for img_key in self.image_keys:
            sk = datum["sequence_key"]
            h5py_image = self.h5py_file[sk][img_key]

            idx = datum[img_key]
            img_arr = np.array(h5py_image[f"image"][idx])
            img_dict[img_key] = img_arr
            curr_ts = (
                h5py_image["header_stamp_secs"][idx][0]
                + h5py_image["header_stamp_nsecs"][idx][0] * 10**-9
            )
            ts.append(curr_ts)

        return img_dict, ts

    def get_long_lat_ele(self, H):
        t = H[:3, 3]
        longitude, latitude = self.XY_To_LonLat(t[0], t[1])
        elevation = t[2]
        return longitude, latitude, elevation

    def get_gridmap_data(self, datum):
        sk = datum["sequence_key"]
        h5py_grid_map = self.h5py_file[sk][self.gridmap_key]
        gm_idx = datum[self.gridmap_key]

        gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

        gps_o_idx = datum[self.gps_o_key]
        gps_p_idx = datum[self.gps_p_key]
        h5py_o_gps = self.h5py_file[sk][self.gps_o_key]
        h5py_p_gps = self.h5py_file[sk][self.gps_p_key]

        h5py_anchor = self.h5py_file[sk][self.anchor_key]
        idx = datum[self.anchor_key]

        elevation_idxs = torch.tensor(
            [
                gm_layers.index(l_name)
                for l_name in self.elevation_layers
                if l_name in gm_layers
            ]
        )

        confidence_idx = gm_layers.index("confidence")

        H_map__sensor_origin_link = get_H_h5py(
            t=h5py_anchor[f"tf_translation_map__sensor_origin_link"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_map__sensor_origin_link"][idx],
        )

        H_utm__map = get_H_h5py(
            t=h5py_anchor[f"tf_translation_utm__map"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_utm__map"][idx],
        )

        longitude, latitude, ele = self.get_long_lat_ele(H_utm__map)

        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
        H_sensor_gravity__map = get_gravity_aligned(H_sensor_origin_link__map)

        H_utm__sensor_origin = get_H_h5py(
            t=h5py_anchor[f"tf_translation_utm__sensor_origin_link"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_utm__sensor_origin_link"][idx],
        )

        long_v, lat_v = self.XY_To_LonLat(
            H_utm__sensor_origin[0, 3], H_utm__sensor_origin[1, 3]
        )

        print(f"\nLat_v: {lat_v}, Long_v: {long_v}, ele_v: {H_utm__sensor_origin[2,3]}")

        ##XXXXXXXXXXXXXXXXXX NEW GPS

        lat_gps = h5py_p_gps["latitude"][gps_o_idx]
        long_gps = h5py_p_gps["longitude"][gps_o_idx]
        ele_gps = h5py_p_gps["altitude"][gps_o_idx]

        orientation = h5py_o_gps["rotation_xyzw"][gps_p_idx]
        rotation_matrix = R.from_quat(orientation).as_matrix()
        euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
        heading_degrees = euler_angles[2]

        print(f"Heading degrees: {heading_degrees}")
        print(f"\nLat_gps: {lat_gps}, Long_gps: {long_gps}, ele_v: {ele_gps}")

        xx, yy = self.transformer.transform(long_gps, lat_gps)
        row, col = self.dem.index(xx, yy)

        wg = self.width_gridmap / 2.0
        crop_polygon = Polygon(
            (
                (xx - wg, yy + wg),
                (xx + wg, yy + wg),
                (xx + wg, yy - wg),
                (xx - wg, yy - wg),
            )
        )
        crop_polygon_rot = rotate(crop_polygon, -1 * heading_degrees, (xx, yy))
        out_image, out_transform = rasterio.mask.mask(
            self.dem, [crop_polygon_rot], crop=True
        )

        # print(f"\nLat: {latitude}, Long: {longitude}, elevation: {ele}")
        # print(H_sensor_gravity__map.inverse()[:2, 3])
        # print(H_map__sensor_origin_link[:2, 3])
        ypr = R.from_matrix(np.array(H_utm__map[:3, :3])).as_euler(
            seq="zyx", degrees=True
        )
        print("yaw pitch roll is ", ypr)
        # yaw_sat = -59
        # yaw_sat = 92.36
        yaw_sat = ypr[0]
        # print(f"ypr is {ypr}")
        # XY_utm__sensor_origin = H_utm__map[:2, 3] + H_map__sensor_origin_link[:2, 3]
        # long_v, lat_v = self.XY_To_LonLat(XY_utm__sensor_origin[0], XY_utm__sensor_origin[1])

        # XY_utm__so = np.array(H_utm__map[:2, 3]) + R_utm__map[:2, :2] @ np.array(H_map__sensor_origin_link[:2, 2])
        # print(XY_utm__so)
        # long_v_u, lat_v_u = self.XY_To_LonLat(XY_utm__so[0], XY_utm__so[1])

        # print(f"\nLat_v: {lat_v}, Long_v: {long_v}")
        # print(f"\nLat_v: {lat_v_u}, Long_v: {long_v_u}")
        out_image[out_image < 5] = np.max(out_image)

        img_arr = np.squeeze(np.array(out_image))

        # img_arr = img_arr.transpose().astype(np.float32)
        # img_arr = img_arr[::-1, ::-1].copy()

        grid_map_data_sat = torch.from_numpy((img_arr))

        # grid_map_data_sat = np.array(grid_map_data_sat)
        # grid_map_data_sat = np.flip(grid_map_data_sat, 0)
        # grid_map_data_sat = np.flip(grid_map_data_sat, 1)

        # grid_map_data_sat = torch.from_numpy((grid_map_data_sat))

        print(f"elevation of GT Map at base {ele_gps}")
        grid_map_data_sat = grid_map_data_sat - ele_gps

        # grid_map_data_sat_rotated = affine(
        #     grid_map_data_sat[None],
        #     angle=yaw_sat,
        #     translate=[0,0],
        #     scale=1,
        #     shear=0,
        #     center=(grid_map_data_sat.shape[0]//2, grid_map_data_sat.shape[1]//2),
        #     fill=torch.nan,
        # )[0]

        grid_map_data_sat_rotated = affine(
            grid_map_data_sat[None],
            angle=-heading_degrees,
            translate=[0, 0],
            scale=1,
            shear=0,
            center=(grid_map_data_sat.shape[0] // 2, grid_map_data_sat.shape[1] // 2),
            fill=torch.nan,
        )[0]

        grid_map_data_sat_rotated = center_crop(
            grid_map_data_sat_rotated, [self.crop_gridmap, self.crop_gridmap]
        )

        #########################################################

        H_map__grid_map_center = torch.eye(4)
        H_map__grid_map_center[:3, 3] = torch.tensor(h5py_grid_map[f"position"][gm_idx])

        H_sensor_gravity__grid_map_center = (
            H_sensor_gravity__map @ H_map__grid_map_center
        )
        pose = H_sensor_gravity__grid_map_center[:2, 3]

        grid_map_resolution = torch.tensor(h5py_grid_map["resolution"][0])
        yaw = R.from_matrix(
            H_sensor_gravity__grid_map_center.clone().numpy()[:3, :3]
        ).as_euler(seq="zyx", degrees=False)[0]
        shift = (H_sensor_gravity__grid_map_center[:2, 3]) / grid_map_resolution
        sh = [shift[1], shift[0]]

        np_data = np.array(h5py_grid_map[f"data"][gm_idx])  # [gm_idx]{gm_idx}
        H_c, W_c = int(np_data.shape[1] / 2), int(np_data.shape[2] / 2)

        grid_map_data = torch.from_numpy(
            np.ascontiguousarray(np.ascontiguousarray(np_data))
        )

        grid_map_data[elevation_idxs[0]][grid_map_data[elevation_idxs[0]] == 0] = np.nan

        grid_map_data[elevation_idxs] = (
            grid_map_data[elevation_idxs] + H_sensor_gravity__grid_map_center[2, 3]
        )

        grid_map_data_rotated = affine(
            grid_map_data[None],
            angle=-np.rad2deg(yaw),
            translate=sh,
            scale=1,
            shear=0,
            center=(H_c, W_c),
            fill=torch.nan,
        )[0]

        grid_map_data_rotated = center_crop(grid_map_data_rotated, (512, 512))

        ts = (
            h5py_grid_map[f"header_stamp_secs"][gm_idx]
            + h5py_grid_map["header_stamp_nsecs"][gm_idx] * 10**-9
        )

        grid_map_data_rotated = np.array(grid_map_data_rotated)
        ###########################################################
        # new_size = (np_data.shape[1], np_data.shape[2])
        # # Use torch.nn.functional.interpolate to resize the tensor
        # grid_map_data_sat_rotated = F.interpolate(grid_map_data_sat_rotated.unsqueeze(0).unsqueeze(0), size=new_size, mode='bilinear', align_corners=True)
        # # Squeeze the tensor to remove the batch and channel dimensions
        # grid_map_data_sat_rotated = grid_map_data_sat_rotated.squeeze()

        elevation_values = np.array(grid_map_data_sat_rotated[:, :])

        # # Create a 3D grid using meshgrid
        x, y = np.meshgrid(np.arange(-72, 72), np.arange(-72, 72))

        elevation_gt = grid_map_data_rotated[elevation_idxs]
        confidence = grid_map_data_rotated[confidence_idx]

        # # elevation_gt = elevation_gt - np.array(H_sensor_gravity__map[2, 3])
        ele_veh_gt = elevation_gt[
            elevation_gt.shape[0] // 2, elevation_gt.shape[1] // 2
        ]
        ele_veh_sat = elevation_values[
            elevation_values.shape[0] // 2, elevation_values.shape[1] // 2
        ]
        # print(f"Offset at vehicle location GT: {ele_veh_gt}")
        # print(f"Offset at vehicle location Sat: {ele_veh_sat}")

        x_gt, y_gt = np.meshgrid(
            np.arange(0, elevation_gt.shape[1]), np.arange(0, elevation_gt.shape[0])
        )

        # elevation_values = elevation_values - (ele_veh_sat - ele_veh_gt)
        elevation_values = elevation_values - (ele_veh_sat - ele_veh_gt)
        elevation_values = elevation_values[::-1, ::-1]
        # print(elevation_values.shape, elevation_gt.shape)
        # error_plot = elevation_values - elevation_gt
        # error_plot[confidence < 0.5] = 0
        # # elevation_values[elevation_gt == 0] = 0
        # # elevation_gt[confidence < 0.5] = 0
        # # idxs = confidence < 0.5
        # idxs = elevation_gt == 0
        # # elevation_gt[idxs] = elevation_values[idxs]
        # print(error_plot.shape)

        gridmap1 = elevation_gt
        gridmap2 = elevation_values
        # gridmap1[elevation_gt==0] = np.nan
        # # gridmap2 = elevation_values + np.array(H_sensor_gravity__grid_map_center[2, 3])
        # # gridmap2 = grid_map_data_sat_rotated2[2:-2, 2:-2]
        # gridmap2 = elevation_values

        ############### ICP ###################
        pointcloud1 = gridmap_to_pointcloud(gridmap1)
        pointcloud2 = gridmap_to_pointcloud(gridmap2, multiplier=1)
        source = pointcloud2
        target = pointcloud1

        voxel_size = 1
        source_down, source_fpfh = preprocess_point_cloud(
            source, voxel_size, return_features=True
        )
        target_down, target_fpfh = preprocess_point_cloud(
            target, voxel_size, return_features=True
        )
        result_ransac = execute_icp(source_down, target_down, voxel_size)
        # draw_registration_result(source_down, target_down, result_ransac.transformation)
        # result_ransac = execute_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        print(result_ransac.transformation)
        print(result_ransac)

        result_tf = np.array(result_ransac.transformation)
        result_ypr = R.from_matrix(result_tf[:3, :3]).as_euler("xyz", degrees=True)
        print(f"RPY (degrees) is {result_ypr} \n Translation is {result_tf[:3,3]}")
        # draw_registration_result(source_down, target_down, result_ransac.transformation)

        pointcloud2 = pointcloud2.transform(result_ransac.transformation)
        gridmap2_new = pointcloud_to_gridmap(
            pointcloud2, multiplier=1, grid_shape=(120, 120)
        )

        gridmap2_new = F.interpolate(
            torch.from_numpy(gridmap2_new).unsqueeze(0).unsqueeze(0),
            scale_factor=5,
            mode="bilinear",
            align_corners=True,
        )
        gridmap2_new = center_crop(gridmap2_new, output_size=gridmap1.shape)
        gridmap2_new = np.array(gridmap2_new.squeeze(0).squeeze(0))

        ############### ICP END ###################

        ############## FUSION #############################

        idxs = confidence > 0.5
        gridmap2_new[idxs] = gridmap1[idxs]

        ###########################################
        # # Create a 3D plot with two subplots
        # fig = plt.figure(figsize=(12, 12))

        # # Subplot 1
        # ax1 = fig.add_subplot(221, projection='3d')
        # ax1.plot_surface(x, y, elevation_values, cmap='viridis')
        # ax1.set_title('Elevation Plot 1')
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_zlabel('Elevation')
        # # ax1.set_zlim(-2, 2)

        # # Subplot 2
        # ax2 = fig.add_subplot(222, projection='3d')
        # ax2.plot_surface(x_gt, y_gt, elevation_gt, cmap='viridis')
        # ax2.set_title('Elevation Plot 2')
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # ax2.set_zlabel('Elevation GT')
        # # ax2.set_zlim(-2, 2)

        # # ax3 = fig.add_subplot(223, projection='3d')
        # # ax3.plot_surface(x, y, error_plot, cmap='viridis')
        # # # ax3.set_zlim(-2, 2)

        # # Set the view to top view
        # ax1.view_init(elev=90, azim=0)
        # ax2.view_init(elev=90, azim=0)
        # # ax3.view_init(elev=90, azim=0)
        # # # Adjust layout to prevent overlapping
        # # plt.tight_layout()

        # # # Show the plots
        # # plt.show()
        # # np_data[elevation_idxs[0]] = np.array(elevation_gt)
        # # np_data = np.concatenate([np_data, np.expand_dims(elevation_values, axis=0)])
        # # gm_layers.append("elevation_satellite")

        grid_map_data_sat = torch.from_numpy(
            np.ascontiguousarray(np.ascontiguousarray(gridmap2_new))
        ).unsqueeze(0)
        gm_sat_layers = ["elevation_sat"]

        ###########################################################

        return (
            np.array(grid_map_data_rotated),
            H_sensor_gravity__map,
            pose,
            yaw,
            np.array(h5py_grid_map["resolution"]),
            gm_layers,
            ts,
            np.array(grid_map_data_sat),
            gm_sat_layers,
        )

    def get_pointcloud_data(self, datum, H_sensor_gravity__map):
        sk = datum["sequence_key"]
        h5py_pointcloud = self.h5py_file[sk][self.pointcloud_key]

        try:
            idx_pointcloud = datum[self.pointcloud_key][-1]
        except:
            idx_pointcloud = datum[self.pointcloud_key]

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

    def get_gvomcloud_data(self, datum, H_sensor_gravity__map):
        sk = datum["sequence_key"]
        h5py_gvom = self.h5py_file[sk][self.gvom_key]
        idx_pointcloud = datum[self.gvom_key]

        H_map__base_link = get_H_h5py(
            t=h5py_gvom[f"tf_translation"][idx_pointcloud],  # {idx_pointcloud}
            q=h5py_gvom[f"tf_rotation_xyzw"][idx_pointcloud],
        )

        valid_point = np.array(h5py_gvom[f"valid"][idx_pointcloud]).sum()
        x = h5py_gvom[f"x"][idx_pointcloud][:valid_point]
        y = h5py_gvom[f"y"][idx_pointcloud][:valid_point]
        z = h5py_gvom[f"z"][idx_pointcloud][:valid_point]
        points = np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)

        H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
        points_sensor_origin = (H_sensor_gravity__base_link.numpy() @ points.T).T

        points_sensor_origin = points_sensor_origin[:, :3]
        ts = (
            h5py_gvom[f"header_stamp_secs"][idx_pointcloud]
            + h5py_gvom["header_stamp_nsecs"][idx_pointcloud] * 10**-9
        )

        return points_sensor_origin, ts

    def get_item(self, idx):
        datum = self.dataset_config[idx]

        image_data, ts_imgs = self.get_images(datum)
        (
            gridmap_data,
            H_sensor_gravity_map,
            pose_grid,
            yaw_grid,
            grid_res,
            grid_layers,
            ts_gridmap,
            gridmap_sat_data,
            grid_sat_layers,
        ) = self.get_gridmap_data(datum)
        pcd_data, ts_pcd = self.get_pointcloud_data(datum, H_sensor_gravity_map)
        gvom_data, ts_gvom = self.get_gvomcloud_data(datum, H_sensor_gravity_map)

        for key in self.image_keys:
            self.vis.image(image_data[key], image_key=key, reference_frame=key)

        # Publish the transform with specific translation and rotation
        self.tf_broadcaster.sendTransform(
            (pose_grid[0], pose_grid[1], 0),
            tf.transformations.quaternion_from_euler(0, 0, yaw_grid),
            rospy.Time.now(),
            "grid_map_frame",
            "sensor_gravity",
        )

        self.vis.pointcloud(pcd_data, reference_frame="sensor_gravity")
        self.vis.gvomcloud(gvom_data, reference_frame="sensor_gravity")
        # self.vis.gridmap_arr(
        #     gridmap_data[:, 1:-1, 1:-1],
        #     res=grid_res,
        #     x=0,
        #     y=0,
        #     layers=grid_layers,
        #     reference_frame="grid_map_frame",
        # )
        self.vis.gridmap_arr(
            gridmap_data[:, 1:-1, 1:-1],
            res=grid_res,
            x=0,
            y=0,
            layers=grid_layers,
            reference_frame="sensor_gravity",
        )

        self.vis.gridmap_arr_sat(
            gridmap_sat_data[:, 1:-1, 1:-1],
            res=grid_res,
            x=0,
            y=0,
            layers=grid_sat_layers,
            reference_frame="sensor_gravity",
        )

        # Publish the transform with specific translation and rotation
        self.tf_broadcaster.sendTransform(
            (pose_grid[0], pose_grid[1], 0),
            tf.transformations.quaternion_from_euler(0, 0, yaw_grid),
            rospy.Time.now(),
            "grid_map_frame",
            "sensor_gravity",
        )

        print("")
        print(f"left img ts diff {ts_imgs[0] - ts_imgs[1]}")
        print(f"right img ts diff {ts_imgs[0] - ts_imgs[2]}")
        print(f"back img ts diff {ts_imgs[0] - ts_imgs[3]}")
        print(f"gridmap trav ts diff {ts_imgs[0] - ts_gridmap}")
        print(f"pointcloud ts diff {ts_imgs[0] - ts_pcd}")
        print(f"GVOM cloud ts diff {ts_imgs[0] - ts_gvom}")
        rospy.sleep(0.1)

        return idx


def signal_handler(sig, frame):
    rospy.signal_shutdown("Ctrl+C detected")
    sys.exit(0)


# Callback functions for keyboard events
current_index = 75


def on_press(key):
    global current_index
    if key == keyboard.Key.right:
        current_index = (current_index + 10) % visualizer.len()
        print("Item: ", visualizer.get_item(current_index))
    elif key == keyboard.Key.left:
        current_index = (current_index - 10) % visualizer.len()
        print("Item: ", visualizer.get_item(current_index))


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--h5_data", type=str, default="", help="Path to h5py file"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="", help="Path to dataset config file"
    )

    print(
        "Use the left (<-) and right (->) arrow keys to iterate through the items of the dataset"
    )

    args = parser.parse_args()

    visualizer = GTVisualization(args.h5_data, args.config)

    # Start listener for keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
