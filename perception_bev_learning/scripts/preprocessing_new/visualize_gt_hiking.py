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
import pickle
import torch
import numpy as np
from pyquaternion import Quaternion  # w, x, y, z
from scipy.spatial.transform import Rotation as R
from torch import from_numpy as fn

from pynput import keyboard
from torchvision.transforms.functional import rotate, affine, center_crop
import tf

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, Image
from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import TransformStamped
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation
import open3d as o3d

"""
Publish GridMap
Publish TF
Publish CameraInfo + Image ?
Publish Maybe Depth Image from Reprojection ?
"""


class SimpleNumpyToRviz:
    def __init__(
        self,
        init_node=True,
        postfix="",
        cv_bridge=None,
        image_keys=[""],
        depth_keys=[""],
        node_name="numpy_to_rviz",
    ):
        if init_node:
            rospy.init_node(node_name, anonymous=False)

        self.cv_bridge = cv_bridge
        self.pub_gvomcloud = rospy.Publisher(
            f"~gvom{postfix}", PointCloud2, queue_size=1
        )
        self.pub_pointcloud = rospy.Publisher(
            f"~pointcloud{postfix}", PointCloud2, queue_size=1
        )
        self.pub_gridmap = rospy.Publisher(f"~gridmap{postfix}", GridMap, queue_size=1)
        self.pub_gridmap_sat = rospy.Publisher(f"~gridmap_sat", GridMap, queue_size=1)
        self.pub_images = {}
        for i in image_keys:
            self.pub_images[i] = rospy.Publisher(
                f"~image{postfix}_{i}", Image, queue_size=1
            )
        self.pub_depth = {}
        for i in depth_keys:
            self.pub_depth[i] = rospy.Publisher(
                f"~image{postfix}_{i}", Image, queue_size=1
            )

        self.pub_camera_info = rospy.Publisher(
            f"~camera_info{postfix}", CameraInfo, queue_size=1
        )
        self.br = tf2_ros.TransformBroadcaster()

    def camera_info(self, camera_info):
        pass

    def tf(self, msg, reference_frame="crl_rzr/map"):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = reference_frame
        t.child_frame_id = msg[3]
        t.transform.translation.x = msg[4][0]
        t.transform.translation.y = msg[4][1]
        t.transform.translation.z = msg[4][2]
        t.transform.rotation.x = msg[5][0]
        t.transform.rotation.y = msg[5][1]
        t.transform.rotation.z = msg[5][2]
        t.transform.rotation.w = msg[5][3]
        self.br.sendTransform(t)

    def image(self, img, reference_frame, image_key=""):
        msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        msg.header.frame_id = reference_frame
        msg.header.stamp = rospy.Time.now()
        self.pub_images[image_key].publish(msg)

    def depth_image(self, img, reference_frame, image_key=""):
        msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        msg.header.frame_id = reference_frame
        msg.header.stamp = rospy.Time.now()
        self.pub_depth[image_key].publish(msg)

    def pointcloud(self, points, reference_frame="sensor_gravity"):
        data = np.zeros(
            points.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        msg = ros_numpy.msgify(PointCloud2, data)

        msg.header.frame_id = reference_frame

        self.pub_pointcloud.publish(msg)

    def gvomcloud(self, points, reference_frame="sensor_gravity"):
        data = np.zeros(
            points.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        msg = ros_numpy.msgify(PointCloud2, data)

        msg.header.frame_id = reference_frame

        self.pub_gvomcloud.publish(msg)

    def gridmap(self, msg, publish=True):
        data_in = msg[0]

        size_x = data_in["data"].shape[1]
        size_y = data_in["data"].shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        layers = []
        data = []

        for i in range(data_in["data"].shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = data_in["data"][i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.position.x = data_in["position"][0]
        info.pose.position.y = data_in["position"][1]
        info.pose.position.z = data_in["position"][2]
        info.pose.orientation.x = data_in["orientation_xyzw"][0]
        info.pose.orientation.y = data_in["orientation_xyzw"][1]
        info.pose.orientation.z = data_in["orientation_xyzw"][2]
        info.pose.orientation.w = data_in["orientation_xyzw"][3]
        info.header = msg[1]
        # info.header.stamp.secs = msg[2]vis
        # info.header.stamp = rospy.Time.now()
        info.resolution = data_in["resolution"]
        info.length_x = size_x * data_in["resolution"]
        info.length_y = size_y * data_in["resolution"]

        gm_msg = GridMap(
            info=info,
            layers=data_in["layers"],
            basic_layers=data_in["basic_layers"],
            data=data,
        )

        if publish:
            self.pub_gridmap.publish(gm_msg)

        return gm_msg

    def gridmap_arr(
        self, arr, res, layers, reference_frame="sensor_gravity", publish=True, x=0, y=0
    ):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(arr.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = arr[i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = rospy.Time.now()
        info.resolution = res
        info.length_x = size_x * res
        info.length_y = size_y * res
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y

        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data)
        if publish:
            self.pub_gridmap.publish(gm_msg)
        return gm_msg

    def gridmap_arr_sat(
        self, arr, res, layers, reference_frame="sensor_gravity", publish=True, x=0, y=0
    ):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        data_dim_0 = MultiArrayDimension()
        data_dim_0.label = "column_index"  # y dimension
        data_dim_0.size = size_y  # number of columns which is y
        data_dim_0.stride = size_y * size_x  # rows*cols
        data_dim_1 = MultiArrayDimension()
        data_dim_1.label = "row_index"  # x dimension
        data_dim_1.size = size_x  # number of rows which is x
        data_dim_1.stride = size_x  # number of rows
        data = []

        for i in range(arr.shape[0]):
            data_tmp = Float32MultiArray()
            data_tmp.layout.dim.append(data_dim_0)
            data_tmp.layout.dim.append(data_dim_1)
            data_tmp.data = arr[i, ::-1, ::-1].transpose().ravel()
            data.append(data_tmp)

        info = GridMapInfo()
        info.pose.orientation.w = 1
        info.header.seq = 0
        info.header.stamp = rospy.Time.now()
        info.resolution = res
        info.length_x = size_x * res
        info.length_y = size_y * res
        info.header.frame_id = reference_frame
        info.pose.position.x = x
        info.pose.position.y = y

        gm_msg = GridMap(info=info, layers=layers, basic_layers=[], data=data)
        if publish:
            self.pub_gridmap_sat.publish(gm_msg)
        return gm_msg
    
def load_pkl(path: str) -> dict:
    with open(path, "rb") as file:
        res = pickle.load(file)
    return res

def get_rot(q_ros):
    return torch.from_numpy(
        Quaternion([q_ros[3], q_ros[0], q_ros[1], q_ros[2]]).rotation_matrix
    ).type(torch.float64)


def get_H_h5py(t, q):
    H = torch.eye(4)
    H[:3, 3] = torch.tensor(t).type(torch.float32)
    H[:3, :3] = get_rot(q)
    return H


def get_H(tf, offset=4):
    H = torch.eye(4)
    H[:3, 3] = torch.tensor(tf[offset]).type(torch.float32)
    H[:3, :3] = get_rot(tf[offset + 1])
    return H


def inv(H):
    H_ = H.clone()
    H_[:3, :3] = H.T[:3, :3]
    H_[:3, 3] = -H.T[:3, :3] @ H[:3, 3]
    return H_


def get_gravity_aligned(H_f__map):
    ypr = R.from_matrix(H_f__map.clone().cpu().numpy()[:3, :3]).as_euler(
        seq="zyx", degrees=True
    )
    H_g__map = H_f__map.clone()
    H_delta = torch.eye(4)

    ypr[0] = 0
    H_delta[:3, :3] = fn(R.from_euler(seq="zyx", angles=ypr, degrees=True).as_matrix())
    H_g__map = inv(H_delta) @ H_g__map

    return H_g__map


def invert_se3(T):
    T_inv = torch.zeros_like(T)
    T_inv[3, 3] = 1.0
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


class GTVisualization:
    def __init__(self, h5py_path, cfg_file):
        self._bridge = CvBridge()

        # (TODO) Can potentially shift the keys to yaml file
        self.image_keys = [
            "hdr_front",
        ]
        # self.gridmap_key = "g_traversability_map_micro"
        self.gridmap_key = "traversability_map_micro"
        self.gridmap_short_key = "traversability_map_short"
        self.anchor_key = "hdr_front"
        self.elevation_layers = ["elevation"]
        self.pointcloud_key = "velodyne_merged_points"

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
        # self.dataset_config = [d for d in self.dataset_config if d["dem_exists"] == True]

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
            t=h5py_anchor[f"tf_translation_map_o3d_localization_manager__base"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_map_o3d_localization_manager__base"][idx],
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

        # grid_map_data_rotated = center_crop(
        #             grid_map_data_rotated, (512, 512)
        #         )
        # grid_map_data_rotated = grid_map_data
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

    def get_gridmap_short_data(self, datum):
        sk = datum["sequence_key"]
        h5py_grid_map = self.h5py_file[sk][self.gridmap_short_key]
        gm_idx = datum[self.gridmap_short_key]

        gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

        np_data = np.array(h5py_grid_map[f"data"][gm_idx])  # [gm_idx]{gm_idx}

        grid_map_data = torch.from_numpy(
            np.ascontiguousarray(np.ascontiguousarray(np_data))
        )

        return (
            np.array(grid_map_data),
            np.array(h5py_grid_map["resolution"]),
            gm_layers,
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
        ) = self.get_gridmap_data(datum)
        # pcd_data, ts_pcd = self.get_raw_pcd_data(datum, H_sensor_gravity_map, 1)
        pcd_data, ts_pcd = self.get_pointcloud_data(datum, H_sensor_gravity_map)

        (
            gridmap_short_data,
            grid_res_short,
            grid_layers_short,
        ) = self.get_gridmap_short_data(datum)

        for key in self.image_keys:
            self.vis.image(image_data[key], image_key=key, reference_frame=key)

        self.vis.pointcloud(pcd_data, reference_frame="sensor_gravity")

        self.vis.gridmap_arr_sat(
            gridmap_short_data[:, 1:-1, 1:-1],
            res=grid_res_short,
            x=0,
            y=0,
            layers=grid_layers_short,
            reference_frame="sensor_gravity",
        )

        self.vis.gridmap_arr(
            gridmap_data[:, 1:-1, 1:-1],
            res=grid_res,
            x=0,
            y=0,
            layers=grid_layers,
            reference_frame="sensor_gravity",
        )

        print("")
        rospy.sleep(0.1)
        print(f"gridmap trav ts diff {ts_imgs[0] - ts_gridmap}")
        print(f"pointcloud ts diff {ts_imgs[0] - ts_pcd}")
        

        return idx


def signal_handler(sig, frame):
    rospy.signal_shutdown("Ctrl+C detected")
    sys.exit(0)


# Callback functions for keyboard events
current_index = 0 # 167 # 1680


def on_press(key):
    global current_index
    if key == keyboard.Key.right:
        current_index = (current_index + 5) % visualizer.len()
        print("Item: ", visualizer.get_item(current_index))
    elif key == keyboard.Key.left:
        current_index = (current_index - 5) % visualizer.len()
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
