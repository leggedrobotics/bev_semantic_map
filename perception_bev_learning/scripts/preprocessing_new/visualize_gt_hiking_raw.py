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

import tf2_ros
import geometry_msgs.msg
from mpl_toolkits.mplot3d import Axes3D
import copy
from utils.tf_utils import get_yaw_oriented_sg, connected_component_search
from scipy import ndimage
from matplotlib import colormaps
from collections import deque
from icp_register import preprocess_point_cloud, execute_icp, draw_registration_result, execute_global_registration
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
        self, arr, res, layers, reference_frame="base", publish=True, x=0, y=0
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

    def gridmap_large(
        self, arr, res, layers, reference_frame="base", publish=True, x=0, y=0
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
    
    def publish_tf(self,  tf_msg):
        self.br.sendTransform(tf_msg)


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

def matrix_to_transform(matrix):
    t = geometry_msgs.msg.TransformStamped()
    t.transform.translation.x = matrix[0, 3]
    t.transform.translation.y = matrix[1, 3]
    t.transform.translation.z = matrix[2, 3]
    
    # Convert rotation matrix to quaternion
    rotation_matrix = matrix[:3, :3]
    quaternion = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1])))
    t.transform.rotation.x = quaternion[0]
    t.transform.rotation.y = quaternion[1]
    t.transform.rotation.z = quaternion[2]
    t.transform.rotation.w = quaternion[3]
    
    return t

class GTVisualization:
    def __init__(self, h5py_path, cfg_file):
        self._bridge = CvBridge()

        # (TODO) Can potentially shift the keys to yaml file
        self.image_keys = [
            "hdr_front",
            "hdr_back",
        ]
        # self.gridmap_key = "g_traversability_map_micro"
        self.gridmap_key = "traversability_map_micro"
        self.gridmap_short_key = "g_traversability_map_short_gt"
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
        # self.dataset_config = [d for d in self.dataset_config if d["is_valid"] == True]

        self.length = len(self.dataset_config)
        self.index_mapping = np.arange(0, self.length)
        # Create a transform broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()

        # print(self.dataset_config)

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

    def get_gridmap_data(self, datum, gridmap_key, crop_dim=None):
        sk = datum["sequence_key"]
        h5py_grid_map = self.h5py_file[sk][gridmap_key]
        gm_idx = datum[gridmap_key]

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
        
        H_map__footprint = get_H_h5py(
            t=h5py_anchor[f"tf_translation_map_o3d_localization_manager__footprint"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_map_o3d_localization_manager__footprint"][idx],
        )

        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
        H_sensor_gravity__map = get_gravity_aligned(H_sensor_origin_link__map)
        # H_map__sensor_gravity = torch.eye(4)
        # H_map__sensor_gravity[:3,:3] = H_map__footprint[:3,:3]
        # H_map__sensor_gravity[:3,3] = H_map__sensor_origin_link[:3, 3]

        # H_sensor_gravity__map = inv(H_map__sensor_gravity)
        H_map__sensor_gravity = inv(H_sensor_gravity__map)

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
        
        if crop_dim is not None:
            grid_map_data_rotated = center_crop(grid_map_data_rotated, crop_dim)
            
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
            H_map__sensor_origin_link,
            H_map__footprint,
            H_map__sensor_gravity
        )
    
    def get_gridmap_data_sensor_gravity(self, datum, gridmap_key, crop_dim=None):
        sk = datum["sequence_key"]
        h5py_grid_map = self.h5py_file[sk][gridmap_key]
        gm_idx = datum[gridmap_key]

        gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

        h5py_anchor = self.h5py_file[sk][self.anchor_key]
        idx = datum[self.anchor_key]

        elevation_idxs = np.array(
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
        
        H_map__footprint = get_H_h5py(
            t=h5py_anchor[f"tf_translation_map_o3d_localization_manager__footprint"][idx],
            q=h5py_anchor[f"tf_rotation_xyzw_map_o3d_localization_manager__footprint"][idx],
        )

        H_sensor_origin_link__map = inv(H_map__sensor_origin_link)
        # H_sensor_gravity__map = get_gravity_aligned(H_sensor_origin_link__map)
        H_map__sensor_gravity = torch.eye(4)
        H_map__sensor_gravity[:3,:3] = H_map__footprint[:3,:3]
        H_map__sensor_gravity[:3,3] = H_map__sensor_origin_link[:3, 3]
        H_map_sgyaw = get_yaw_oriented_sg(H_map__sensor_gravity)
        H_sensor_gravity__map = inv(H_map__sensor_gravity)
        H_sgyaw__map = inv(H_map_sgyaw)

        # Assuming that the X and Y lengths are the same 
        # Assuming that there is only 1 elevation layer
        resolution = h5py_grid_map["resolution"][0]
        length_x = h5py_grid_map[f"length"][0] // resolution
        length_y = h5py_grid_map[f"length"][1] // resolution

        origin = np.array(h5py_grid_map[f"position"][gm_idx])

        np_data = np.array(h5py_grid_map[f"data"][gm_idx]) 

        elevation_data = np_data[elevation_idxs[0]]

        connected_elevation_map = connected_component_search(elevation_data, threshold=0.5)
        trav_idx = gm_layers.index("navigation_traversability")
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

        points_trav = nav_trav_map[indices]
        trav_indices = points_trav > 0.2


        points_map = np.vstack((x_points, y_points, z_points, np.ones_like(x_points)))
        points_sg = (np.array(H_sgyaw__map) @ points_map).T

        ############################################

        # # DO ICP
        pcd_data, _ = self.get_pointcloud_data(datum, H_sgyaw__map)

        pcd_data_o3d = o3d.geometry.PointCloud()
        pointsg_data_o3d = o3d.geometry.PointCloud()
        pointsg_data_o3d_icp = o3d.geometry.PointCloud()
        # Convert the numpy array to an Open3D Vector3dVector
        pcd_data_o3d.points = o3d.utility.Vector3dVector(pcd_data)
        pointsg_data_o3d.points = o3d.utility.Vector3dVector(points_sg[:,:3])
        pointsg_data_o3d_icp.points = o3d.utility.Vector3dVector(points_sg[trav_indices][:,:3])
        
        voxel_size_o3d = 0.1

        source_down = preprocess_point_cloud(pointsg_data_o3d_icp, 0.2)
        target_down = preprocess_point_cloud(pcd_data_o3d, 0.05)
        result_ransac = execute_icp(
            source_down, target_down, voxel_size_o3d
        )

        # source_down, source_fpfh = preprocess_point_cloud(pointsg_data_o3d, 0.2, return_features=True)
        # target_down, target_fpfh = preprocess_point_cloud(pcd_data_o3d, 0.2, return_features=True)
        # result_ransac = execute_global_registration(source_down, target_down,
        #                         source_fpfh, target_fpfh,
        #                         0.2)
        print(f"ICP converged: {result_ransac.inlier_rmse}")
        print(f"RMSE: {result_ransac.inlier_rmse}")
        print(f"Fitness: {result_ransac.fitness}")
        result_tf = np.array(result_ransac.transformation)
        result_ypr = R.from_matrix(result_tf[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        print(f"RPY (degrees) is {result_ypr} \n Translation is {result_tf[:3,3]}")
        # draw_registration_result(source_down, target_down, result_ransac.transformation)
        pointsg_data_o3d = pointsg_data_o3d.transform(
            result_ransac.transformation
        )

        points_sg = np.asarray(pointsg_data_o3d.points)

        ############################################


        new_shape = (int(length_y * 1.42), int(length_x * 1.42))
        new_grid_map = np.full(new_shape, np.nan)

        additional_fields_values = additional_fields_data[:, indices[0], indices[1]]

        print(f"add field values shape is {additional_fields_values.shape}")

        try:
            new_additional_fields = [np.full(new_shape, np.nan) for _ in range(len(additional_fields_data))]

            
            ########################################
            values = points_sg[:, 2]  # Assuming z-values represent the values in the gridmap

            # Normalize coordinates to match the grid indices
            normalized_coords = (
                (points_sg[:, :2] / resolution) + np.array(new_shape) / 2
            ).astype(int)

            new_grid_map[normalized_coords[:, 0], normalized_coords[:, 1]] = values
            
            for layer_idx in range(len(new_additional_fields)):
                new_additional_fields[layer_idx][normalized_coords[:, 0], normalized_coords[:, 1]] = additional_fields_values[layer_idx]

            grid_map_data_rotated = np.insert(new_additional_fields, elevation_idxs[0], new_grid_map[...,], axis=0)


        except Exception as e:
            print(f"Error occurred: {e}")

        ## Interpoaltion
        # Step 1: Create a mask of NaNs
        nan_mask = np.isnan(grid_map_data_rotated)
        
        # Step 2: Use distance transform to find the nearest non-NaN value
        # Distances and indices of the nearest non-NaN values
        distances, nearest_indices = ndimage.distance_transform_edt(nan_mask, return_indices=True)
        
        # Get the nearest values for NaN positions
        nearest_values = grid_map_data_rotated[tuple(nearest_indices)]
        
        # Step 3: Ensure replacements are only made within the specified max_distance
        within_distance_mask = distances <= 1
        
        # Step 4: Combine the masks and fill NaNs
        fill_mask = nan_mask & within_distance_mask
        filled_image = grid_map_data_rotated.copy()
        filled_image[fill_mask] = nearest_values[fill_mask]

        #######################################

        # H_map__grid_map_center = torch.eye(4)
        # H_map__grid_map_center[:3, 3] = torch.tensor(h5py_grid_map[f"position"][gm_idx])

        # H_sensor_gravity__grid_map_center = (
        #     H_sensor_gravity__map @ H_map__grid_map_center
        # )
        # pose = H_sensor_gravity__grid_map_center[:2, 3]

        # grid_map_resolution = torch.tensor(h5py_grid_map["resolution"][0])
        # yaw = R.from_matrix(
        #     H_sensor_gravity__grid_map_center.clone().numpy()[:3, :3]
        # ).as_euler(seq="zyx", degrees=False)[0]
        # shift = (H_sensor_gravity__grid_map_center[:2, 3]) / grid_map_resolution
        # sh = [shift[1], shift[0]]

        # np_data = np.array(h5py_grid_map[f"data"][gm_idx])  # [gm_idx]{gm_idx}
        # H_c, W_c = int(np_data.shape[1] / 2), int(np_data.shape[2] / 2)

        # grid_map_data = torch.from_numpy(
        #     np.ascontiguousarray(np.ascontiguousarray(np_data))
        # )

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
        
        if crop_dim is not None:
            grid_map_data_rotated = center_crop(torch.Tensor(filled_image)[None], crop_dim)
            
        # grid_map_data_rotated = grid_map_data
        ts = (
            h5py_grid_map[f"header_stamp_secs"][gm_idx]
            + h5py_grid_map["header_stamp_nsecs"][gm_idx] * 10**-9
        )

        return (
            np.array(grid_map_data_rotated[0]),
            H_sgyaw__map,
            None, # pose,
            None, # yaw,
            np.array(h5py_grid_map["resolution"]),
            gm_layers,
            ts,
            H_map__sensor_origin_link,
            H_map__footprint,
            H_map__sensor_gravity,
            points_sg[:,:3],
            H_map_sgyaw
        )

    def get_gridmap_short_data(self, gridmap_key, datum):
        try:
            sk = datum["sequence_key"]
            h5py_grid_map = self.h5py_file[sk][gridmap_key]
            gm_idx = datum[gridmap_key]

            gm_layers = [g.decode("utf-8") for g in h5py_grid_map["layers"]]

            np_data = np.array(h5py_grid_map[f"data"][gm_idx])  # [gm_idx]{gm_idx}

            grid_map_data = torch.from_numpy(
                np.ascontiguousarray(np.ascontiguousarray(np_data))
            )[0]
            H_sgyaw__map = torch.Tensor(h5py_grid_map[f"T_sensor_gravity_yaw__map"][gm_idx])
            return (
                np.array(grid_map_data),
                np.array(h5py_grid_map["resolution"]),
                gm_layers,
                H_sgyaw__map,
            )
        except Exception as e:
            print(f"error occured {e}")
    
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
    
    def project_pointcloud_onto_images(self, datum, pcd_data, H_sensor_gravity__map):

        img_dict = {}
        try:
            for img_key in self.image_keys:
                sk = datum["sequence_key"]
                h5py_image = self.h5py_file[sk][img_key]
                h5py_camera_info = self.h5py_file[sk][f'{img_key}-camera_info']
                idx = datum[img_key]

                img_arr = np.array(h5py_image[f"image"][idx])
                K_mat = np.array(h5py_camera_info["K"]).reshape(3, 3)

                # print(np.array(h5py_camera_info["height"]), np.array(h5py_camera_info["width"]))

                # print(f"K mat is {K_mat}")
                # print(f"Image shape is {img_arr.shape}")

                H_map__camera = get_H_h5py(
                    t=h5py_image[f"tf_translation"][idx],
                    q=h5py_image[f"tf_rotation_xyzw"][idx],
                )
                H_sensor_gravity__camera = np.array(H_sensor_gravity__map @ H_map__camera)

                points = np.stack([pcd_data[:, 0], pcd_data[:, 1], pcd_data[:, 2], np.ones((pcd_data.shape[0],))], axis=1)

                points_camera = (np.linalg.inv(H_sensor_gravity__camera) @ points.T).T

                # Project onto image plane using K
                points_image = (K_mat @ points_camera[:, :3].T).T

                depths = points_image[:, 2]
                points_xy = points_image[:, :2] / points_image[:, 2:3]

                # Apply the maskings
                # Remove points that are either outside or behind the camera.
                # Leave a margin of 1 pixel for aesthetic reasons. Also make
                # sure points are at least 1m in front of the camera to avoid
                # seeing the lidar points on the camera casing for non-keyframes
                # which are slightly out of sync.

                mask = np.ones(depths.shape[0], dtype=bool)
                mask = np.logical_and(mask, depths > 0.3)
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
                normalized_depth = (depths - np.min(depths)) / (20 - np.min(depths))
                # normalized_depth = np.log(depths) / 5

                # Choose a colormap
                cmap = colormaps.get_cmap("viridis")  # You can choose other colormaps

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

                # Overlay the valid projected points on the image with thickness
                for point, color in zip(points_2d, colors):
                    x, y = point
                    cv2.circle(image, (x, y), radius=1, color=color.tolist(), thickness=-1)  # Change radius and thickness as needed

                # image[points_2d[:, 1], points_2d[:, 0]] = colors

                overlay = cv2.addWeighted(img_arr, 0.5, image, 0.5, 0)

                img_dict[img_key] = np.array(overlay)


                # # Save the resulting image in the dictionary
                # img_dict[img_key] = img_arr

        except Exception as e:
            print(f"Error occurred: {e}")

        return img_dict



    def get_item(self, idx):
        datum = self.dataset_config[idx]

        image_data, ts_imgs = self.get_images(datum)
        
        # (
        #     gridmap_data,
        #     H_sensor_gravity_map,
        #     pose_grid,
        #     yaw_grid,
        #     grid_res,
        #     grid_layers,
        #     ts_gridmap,
        #     H_map_base,
        #     H_map_footprint,
        #     H_map_sg
        # ) = self.get_gridmap_data(datum, self.gridmap_key)

       

        # (
        #     gridmap_short_data,
        #     H_sgyaw_map,
        #     pose_grid,
        #     yaw_grid,
        #     grid_res_short,
        #     grid_layers_short,
        #     ts_gridmap_short, 
        #     H_map_base,
        #     H_map_footprint,
        #     H_map_sg,
        #     points_sg, 
        #     H_map_sgyaw
        # ) = self.get_gridmap_data_sensor_gravity(datum, self.gridmap_short_key, crop_dim=(276, 276))
        
        (
            gridmap_short_data,
            H_sgyaw_map,
            pose_grid,
            yaw_grid,
            grid_res_short,
            grid_layers_short,
            ts_gridmap_short, _,_,_
        ) = self.get_gridmap_data(datum, self.gridmap_short_key, crop_dim=(276, 276))

        # pcd_data, ts_pcd = self.get_pointcloud_data(datum, H_sgyaw_map)
        
        # image_data = self.project_pointcloud_onto_images(datum, pcd_data, H_sgyaw_map)
        
        # # Create TransformStamped messages for each transform
        # t_map_base1 = matrix_to_transform(H_map_base)
        # t_map_base2 = matrix_to_transform(H_map_footprint)
        # t_map_base3 = matrix_to_transform(H_map_sg)
        # t_map_base4 = matrix_to_transform(H_map_sgyaw)
        
        # # Set header frames and times
        # t_map_base1.header.stamp = rospy.Time.now()
        # t_map_base1.header.frame_id = "map"
        # t_map_base1.child_frame_id = "base"
        
        # t_map_base2.header.stamp = rospy.Time.now()
        # t_map_base2.header.frame_id = "map"
        # t_map_base2.child_frame_id = "footprint"
        
        # t_map_base3.header.stamp = rospy.Time.now()
        # t_map_base3.header.frame_id = "map"
        # t_map_base3.child_frame_id = "sensor_gravity"

        # t_map_base4.header.stamp = rospy.Time.now()
        # t_map_base4.header.frame_id = "map"
        # t_map_base4.child_frame_id = "sensor_gravity_yaw"

        # # # Publish transforms
        # self.vis.publish_tf(t_map_base1)
        # self.vis.publish_tf(t_map_base2)
        # self.vis.publish_tf(t_map_base3)
        # self.vis.publish_tf(t_map_base4)


        for key in self.image_keys:
            self.vis.image(image_data[key], image_key=key, reference_frame=key)

        # self.vis.pointcloud(pcd_data, reference_frame="sensor_gravity_yaw")
        # self.vis.gvomcloud(points_sg, reference_frame="sensor_gravity_yaw")


        # self.vis.gridmap_arr(
        #     gridmap_data[:, 1:-1, 1:-1],
        #     res=grid_res,
        #     x=0,
        #     y=0,
        #     layers=grid_layers,
        #     reference_frame="sensor_gravity_yaw",
        # )

        self.vis.gridmap_large(
            gridmap_short_data[:, 1:-1, 1:-1],
            res=grid_res_short,
            x=0,
            y=0,
            layers=grid_layers_short,
            reference_frame="sensor_gravity_yaw",
        )

        print("")
        rospy.sleep(0.1)
        # print(f"gridmap trav ts diff {ts_imgs[0] - ts_gridmap}")
        # print(f"pointcloud ts diff {ts_imgs[0] - ts_pcd}")

        return idx


def signal_handler(sig, frame):
    rospy.signal_shutdown("Ctrl+C detected")
    sys.exit(0)


# Callback functions for keyboard events
current_index = 280 # 167 # 1680


def on_press(key):
    global current_index
    if key == keyboard.Key.right:
        current_index = (current_index + 1) % visualizer.len()
        print("Item: ", visualizer.get_item(current_index))
    elif key == keyboard.Key.left:
        current_index = (current_index - 1) % visualizer.len()
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
