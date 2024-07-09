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
from perception_bev_learning.utils import inv, get_gravity_aligned
from perception_bev_learning.ros import SimpleNumpyToRviz
from perception_bev_learning.utils import load_pkl, get_H_h5py
from perception_bev_learning.dataset.h5py_keys import *
from pynput import keyboard

import tf
import geometry_msgs.msg

elevation_layers = ["elevation"]


def get_sequence_key(sk):
    if sk.find("_with_data") != -1:
        # New dataset format
        sk_new = sk[: sk.find("_with_data")]
    return sk_new


class GTVisualization:
    def __init__(self, h5py_path, cfg_file, gridmap_topic=MAP_MICRO):
        self._bridge = CvBridge()
        self.gridmap_topic = gridmap_topic
        self.image_keys = ["image_front", "image_left", "image_right", "image_back"]
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

    def len(self):
        return len(self.index_mapping)

    def get_images(
        self,
        datum,
        cameras=[
            (IMAGE_FRONT, "image_front"),
            (IMAGE_LEFT, "image_left"),
            (IMAGE_RIGHT, "image_right"),
            (IMAGE_BACK, "image_back"),
        ],
    ):
        img_dict = {}

        for h5py_camera_key, img_name in cameras:
            sk = datum["sequence_key"]
            sk_new = get_sequence_key(sk)
            h5py_image = self.h5py_file[sk_new][h5py_camera_key]

            idx = datum[img_name]
            img_arr = np.array(h5py_image[f"image"][idx])
            img_dict[img_name] = img_arr

        return img_dict

    def get_gridmap_data(self, datum):
        topic = self.gridmap_topic
        sk = datum["sequence_key"]
        sk_new = get_sequence_key(sk)
        h5py_grid_map = self.h5py_file[sk_new][topic]
        gm_idx = datum["gridmap_micro"]

        gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]
        h5py_image = self.h5py_file[sk_new][IMAGE_FRONT]
        idx = datum["image_front"]

        elevation_idxs = torch.tensor(
            [
                gm_layers.index(l_name)
                for l_name in elevation_layers
                if l_name in gm_layers
            ]
        )

        H_map__sensor_origin_link = get_H_h5py(
            t=h5py_image[f"tf_translation_map__sensor_origin_link"][idx],
            q=h5py_image[f"tf_rotation_xyzw_map__sensor_origin_link"][idx],
        )
        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
        H_sensor_gravity__map = get_gravity_aligned(H_sensor_origin_link__map)

        H_map__grid_map_center = torch.eye(4)
        H_map__grid_map_center[:3, 3] = torch.tensor(
            h5py_grid_map[f"position"][gm_idx]
        )  # {gm_idx}
        print(f"gridmap pose in map is {H_map__grid_map_center[:3,3]}")
        print(f"sensor gravity origin is {H_sensor_gravity__map.inverse()[:3,3]}")

        grid_map_resolution = torch.tensor(h5py_grid_map["resolution"][0])
        H_sensor_gravity__grid_map_center = (
            H_sensor_gravity__map @ H_map__grid_map_center
        )
        pose = H_sensor_gravity__grid_map_center[:3, 3]
        np_data = np.array(h5py_grid_map[f"data"][gm_idx])
        grid_map_data = torch.from_numpy(
            np.ascontiguousarray(np.ascontiguousarray(np_data))
        )

        grid_map_data[elevation_idxs] = (
            grid_map_data[elevation_idxs] + H_sensor_gravity__grid_map_center[2, 3]
        )

        yaw = R.from_matrix(
            H_sensor_gravity__grid_map_center.clone().numpy()[:3, :3]
        ).as_euler(seq="zyx", degrees=False)[0]
        shift = (H_sensor_gravity__grid_map_center[:2, 3]) / grid_map_resolution
        print(
            f"gridmap center in sensor gravity {H_sensor_gravity__grid_map_center[:3,3]}"
        )

        # yaw = 0
        # pose = H_map__grid_map_center[:2, 3]
        # Publish the transform with specific translation and rotation

        return (
            np.array(grid_map_data),
            H_sensor_gravity__map,
            pose,
            yaw,
            np.array(h5py_grid_map["resolution"]),
            gm_layers,
            H_sensor_gravity__grid_map_center.clone().numpy(),
        )

    def get_pointcloud_data(self, datum, topic, H_sensor_gravity__map):
        sk = datum["sequence_key"]
        sk_new = get_sequence_key(sk)
        h5py_pointcloud = self.h5py_file[sk_new][topic]
        idx_pointcloud = datum["pointclouds"][-1]

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
        # points_sensor_origin = (H_map__base_link.numpy() @ points.T).T
        points_sensor_origin = points_sensor_origin[:, :3]

        return points_sensor_origin

    def get_gvomcloud_data(self, datum, topic, H_sensor_gravity__map):
        sk = datum["sequence_key"]
        sk_new = get_sequence_key(sk)
        h5py_gvomcloud = self.h5py_file[sk_new][topic]
        idx_gvomcloud = datum["gvom_micro"][-1]

        H_map__base_link = get_H_h5py(
            t=h5py_gvomcloud[f"tf_translation"][idx_gvomcloud],  # {idx_pointcloud}
            q=h5py_gvomcloud[f"tf_rotation_xyzw"][idx_gvomcloud],
        )

        valid_point = np.array(h5py_gvomcloud[f"valid"][idx_gvomcloud]).sum()
        x = h5py_gvomcloud[f"x"][idx_gvomcloud][:valid_point]
        y = h5py_gvomcloud[f"y"][idx_gvomcloud][:valid_point]
        z = h5py_gvomcloud[f"z"][idx_gvomcloud][:valid_point]
        points = np.stack([x, y, z, np.ones((x.shape[0],))], axis=1)

        H_sensor_gravity__base_link = H_sensor_gravity__map @ H_map__base_link
        points_sensor_origin = (H_sensor_gravity__base_link.numpy() @ points.T).T
        # points_sensor_origin = (H_map__base_link.numpy() @ points.T).T
        points_sensor_origin = points_sensor_origin[:, :3]

        return points_sensor_origin

    def get_item(self, idx):
        datum = self.dataset_config[idx]

        image_data = self.get_images(datum)
        (
            gridmap_data,
            H_sensor_gravity_map,
            pose_grid,
            yaw_grid,
            grid_res,
            grid_layers,
            H_mat,
        ) = self.get_gridmap_data(datum)
        pcd_data = self.get_pointcloud_data(datum, PCD_MERGED, H_sensor_gravity_map)
        gvom_data = self.get_gvomcloud_data(datum, GVOM_MICRO, H_sensor_gravity_map)

        for key in self.image_keys:
            self.vis.image(image_data[key], image_key=key, reference_frame=key)

        translation = H_mat[:3, 3]
        quaternion = tf.transformations.quaternion_from_matrix(H_mat)

        self.tf_broadcaster.sendTransform(
            (translation[0], translation[1], 0),
            tf.transformations.quaternion_from_euler(0, 0, yaw_grid),
            rospy.Time.now(),
            "grid_map_frame",
            "sensor_gravity",
        )

        self.vis.pointcloud(pcd_data, reference_frame="sensor_gravity")
        self.vis.gvomcloud(gvom_data, reference_frame="sensor_gravity")

        self.vis.gridmap_arr(
            gridmap_data[:, 1:-1, 1:-1],
            res=grid_res,
            x=0,
            y=0,
            layers=grid_layers,
            reference_frame="grid_map_frame",
        )

        self.tf_broadcaster.sendTransform(
            (translation[0], translation[1], 0),
            tf.transformations.quaternion_from_euler(0, 0, yaw_grid),
            rospy.Time.now(),
            "grid_map_frame",
            "sensor_gravity",
        )
        # print(f"X: {pose_grid[0]}, Y:{pose_grid[1]}, Yaw:{yaw_grid}")
        # rospy.sleep(2)
        return idx


def signal_handler(sig, frame):
    rospy.signal_shutdown("Ctrl+C detected")
    sys.exit(0)


# Callback functions for keyboard events
current_index = 60


def on_press(key):
    global current_index
    if key == keyboard.Key.right:
        current_index = (current_index + 1) % visualizer.len()
        print("Next Item:", visualizer.get_item(current_index))
    elif key == keyboard.Key.left:
        current_index = (current_index - 1) % visualizer.len()
        print("Previous Item:", visualizer.get_item(current_index))


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

    args = parser.parse_args()

    visualizer = GTVisualization(args.h5_data, args.config)

    # for i in range(visualizer.len()):
    #     visualizer.get_item(i)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
