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
from torchvision.transforms.functional import rotate, affine, center_crop
import tf
from scripts.preprocessing_new.utils.tf_utils import get_np_T
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

matplotlib.use("TkAgg")  # Use the TkAgg backend (replace with your preferred backend)
import matplotlib.pyplot as plt
from pytictac import Timer


class GTVisualization:
    def __init__(self, h5py_path, cfg_file):
        self._bridge = CvBridge()

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

    def get_gridmap_data(self, datum):
        sk = datum["sequence_key"]
        h5py_grid_map = self.h5py_file[sk][self.gridmap_key]
        gm_idx = datum[self.gridmap_key]

        gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

        h5py_anchor = self.h5py_file[sk][self.anchor_key]
        idx = datum[self.anchor_key]

        elevation_idxs = torch.tensor(
            [
                gm_layers.index(l_name)
                for l_name in self.elevation_layers
                if l_name in gm_layers
            ]
        )

        H_map__sensor_origin_link = get_H_h5py(
            t=h5py_anchor[f"tf_translation_map__sensor_origin_link"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_map__sensor_origin_link"][idx],
        )
        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
        H_sensor_gravity__map = get_gravity_aligned(H_sensor_origin_link__map)

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

        # grid_map_data[elevation_idxs] = (
        #     grid_map_data[elevation_idxs] + H_sensor_gravity__grid_map_center[2, 3]
        # )

        # grid_map_data_rotated = affine(
        #         grid_map_data[None],
        #         angle=-np.rad2deg(yaw),
        #         translate=sh,
        #         scale=1,
        #         shear=0,
        #         center=(H_c, W_c),
        #         fill=torch.nan,
        #     )[0]

        # grid_map_data_rotated = center_crop(
        #             grid_map_data_rotated, (512, 512)
        #         )
        grid_map_data_rotated = grid_map_data
        ts = (
            h5py_grid_map[f"header_stamp_secs"][gm_idx]
            + h5py_grid_map["header_stamp_nsecs"][gm_idx] * 10**-9
        )

        return (
            np.array(grid_map_data_rotated),
            H_sensor_gravity__map,
            pose,
            yaw,
            np.array(h5py_grid_map["resolution"]),
            gm_layers,
            ts,
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

    def get_depth_data(self, datum):
        """
        Function returns the depth map for the images
        """
        img_dict = {}
        min_dist = 1
        sk = datum["sequence_key"]

        h5py_pointcloud = self.h5py_file[sk][self.pointcloud_key]
        with Timer("images"):
            for img_key in self.image_keys:
                # For each Image
                camera_info_key = str(img_key) + "-camera_info"
                h5py_image = self.h5py_file[sk][img_key]
                h5py_camera_info = self.h5py_file[sk][camera_info_key]

                idx = datum[img_key]
                img_arr = np.array(h5py_image[f"image"][idx])

                K = np.array(h5py_camera_info["P"]).reshape(3, 4)[:3, :3]
                # K = np.array(h5py_camera_info["K"]).reshape(3,3)
                # TODO something went wrong with some Camp Roberts DS so correcting here
                if K[0, 2] == 640:
                    K[:2, :] *= 0.5

                H_map__camera = get_np_T(
                    t=h5py_image[f"tf_translation"][idx],
                    q=h5py_image[f"tf_rotation_xyzw"][idx],
                )

                points_baselink = []
                H_array = []
                # Here we will loop through all the pointcloud data indices and transform them to the Map frame
                with Timer("extract pcl data"):
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

                # print(points_baselink.shape)
                # print(H_array.shape)
                with Timer("TF points baselink"):
                    points_map = np.matmul(H_array, points_baselink[:, :, None])[
                        :, :, 0
                    ]

                with Timer("TF to Camera"):
                    # Now we transform them to the current camera frame
                    points_camera = (np.linalg.inv(H_map__camera) @ points_map.T).T

                with Timer("TF to image"):
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

                mask = np.ones(depths.shape[0], dtype=bool)
                mask = np.logical_and(mask, depths > min_dist)
                mask = np.logical_and(mask, points_xy[:, 0] < img_arr.shape[1] - 1)
                mask = np.logical_and(mask, points_xy[:, 0] > 1)
                mask = np.logical_and(mask, points_xy[:, 1] < img_arr.shape[0] - 1)
                mask = np.logical_and(mask, points_xy[:, 1] > 1)
                points_xy = points_xy[mask, :]
                depths = depths[mask]

                sorted_indices = np.argsort(depths)[::-1]
                points_xy = points_xy[sorted_indices]
                depths = depths[sorted_indices]

                # Normalize depth values to [0, 1]
                # normalized_depth = (depths - np.min(depths)) / (100 - np.min(depths))
                normalized_depth = np.log(depths) / 5

                # Choose a colormap
                cmap = cm.get_cmap("viridis")  # You can choose other colormaps

                # Map normalized depth values to colors
                colors = (cmap(normalized_depth) * 255).astype(np.uint8)
                colors = colors[:, :3]
                colors = colors[:, ::-1]
                # Convert 3D points to 2D pixel coordinates
                # Replace this transformation with your actual camera projection logic
                points_2d = np.round(points_xy).astype(int)

                # Create an image with zeros (black background)
                image = np.zeros(
                    (img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8
                )

                image[points_2d[:, 1], points_2d[:, 0]] = colors

                overlay = cv2.addWeighted(img_arr, 0.5, image, 0.5, 0)

                img_dict[img_key] = np.array(overlay)

        return img_dict

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

        image_data = self.get_depth_data(datum)

        (
            gridmap_data,
            H_sensor_gravity_map,
            pose_grid,
            yaw_grid,
            grid_res,
            grid_layers,
            ts_gridmap,
        ) = self.get_gridmap_data(datum)

        pcd_data, ts_pcd = self.get_pointcloud_data(datum, H_sensor_gravity_map)
        gvom_data, ts_gvom = self.get_gvomcloud_data(datum, H_sensor_gravity_map)

        for key in self.image_keys:
            self.vis.image(image_data[key], image_key=key, reference_frame=key)

        self.vis.pointcloud(pcd_data, reference_frame="sensor_gravity")
        self.vis.gvomcloud(gvom_data, reference_frame="sensor_gravity")

        # self.vis.gridmap_arr(
        #     gridmap_data[:, 1:-1, 1:-1],
        #     res=grid_res,
        #     x=0,
        #     y=0,
        #     layers=grid_layers,
        #     reference_frame="sensor_gravity",
        # )

        rospy.sleep(0.1)

        return idx


def signal_handler(sig, frame):
    rospy.signal_shutdown("Ctrl+C detected")
    sys.exit(0)


# Callback functions for keyboard events
current_index = 0


def on_press(key):
    global current_index
    if key == keyboard.Key.right:
        current_index = (current_index + 2) % visualizer.len()
        print("Item: ", visualizer.get_item(current_index))
    elif key == keyboard.Key.left:
        current_index = (current_index - 2) % visualizer.len()
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
    print(args.config)
    visualizer = GTVisualization(args.h5_data, args.config)

    # Start listener for keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
