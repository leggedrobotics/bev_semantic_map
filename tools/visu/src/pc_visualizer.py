#!/usr/bin/env python

"""
Visualizes the point cloud and the grid map.

Author: Robin Schmid
Date: Dec 2023
"""


import os
import sys
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import OccupancyGrid
import sensor_msgs.point_cloud2
import numpy as np
import cv2
import torch

np.set_printoptions(threshold=sys.maxsize)

RESOLUTION = 0.05
LAYERS = ["mask"]


class PcdVisualizer:
    def __init__(self):
        self.show_pc1 = rospy.get_param("show_pc1")
        self.show_pc2 = rospy.get_param("show_pc2")
        self.show_label = rospy.get_param("show_label")
        self.show_pred = rospy.get_param("show_pred")
        self.show_frustrum = rospy.get_param("show_frustrum")

        self.data_dir = rospy.get_param("data_dir")
        self.pred_dir = rospy.get_param("pred_dir")
        
        self.pub_pc1 = rospy.Publisher("point_cloud1", PointCloud2, queue_size=1)
        self.pub_pc2 = rospy.Publisher("point_cloud2", PointCloud2, queue_size=1)
        self.pub_frustrum = rospy.Publisher("frustrum", PointCloud2, queue_size=1)
        self.pub_grid_map = rospy.Publisher("grid_map", GridMap, queue_size=1)
        self.pub_occupancy_map = rospy.Publisher("occupancy_grid", OccupancyGrid, queue_size=1)

    def point_cloud_process1(self, point_cloud, publish=False):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        # Create a PointCloud2 message
        pointcloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_pc1.publish(pointcloud_msg)

    def point_cloud_process2(self, point_cloud, publish=False):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        # Create a PointCloud2 message
        pointcloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_pc2.publish(pointcloud_msg)
    
    def frustrum_process(self, point_cloud, publish=False):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        point_cloud = point_cloud.reshape(-1, 3)

        # Create a PointCloud2 message
        pointcloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_frustrum.publish(pointcloud_msg)

    def correct_z_direction(self, point_cloud):
        # Increase z value by 0.6
        point_cloud[:, 2] += 0.6

        return point_cloud

    def grid_map_arr(self, arr, res, layers, reference_frame="world", publish=True, x=0, y=0):
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
            data_tmp.data = arr[i, ::-1, ::-1].ravel()
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
            self.pub_grid_map.publish(gm_msg)
        return gm_msg

    def occupancy_map_arr(self, arr, res, reference_frame="world", publish=True, x=0, y=0):
        size_x = arr.shape[1]
        size_y = arr.shape[2]

        center_offset_x = (size_x * res) / 2
        center_offset_y = (size_y * res) / 2

        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.seq = 0
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = reference_frame
        occupancy_grid.info.resolution = res
        occupancy_grid.info.width = size_x
        occupancy_grid.info.height = size_y
        occupancy_grid.info.origin.position.x = x - center_offset_x  # Adjusted x origin
        occupancy_grid.info.origin.position.y = y - center_offset_y  # Adjusted y origin
        occupancy_grid.data = (arr * 100).astype(np.int8).ravel()  # Scale values to 0-100 and flatten

        if publish:
            self.pub_occupancy_map.publish(occupancy_grid)
        return occupancy_grid


if __name__ == "__main__":
    rospy.init_node("bevnet_visualizer", anonymous=False)

    vis = PcdVisualizer()

    num_files = 0

    if vis.show_pc1:
        pc_dir1 = os.path.join(vis.data_dir, "pcd")
        pc_files1 = sorted([f for f in os.listdir(pc_dir1) if f.endswith(".pt")])

        if len(pc_files1) > 0:
            num_files = len(pc_files1)
        else:
            vis.show_pc1 = False

    if vis.show_pc2:
        pc_dir2 = os.path.join(vis.data_dir, "pcd_single")
        pc_files2 = sorted([f for f in os.listdir(pc_dir2) if f.endswith(".pt")])

        if len(pc_files2) > 0:
            num_files = len(pc_files2)
        else:
            vis.show_pc2 = False

    if vis.show_label:
        label_dir = os.path.join(vis.data_dir, "bin_trav")
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".pt")])

        if len(label_files) > 0:
            num_files = len(label_files)
        else:
            vis.show_label = False

    if vis.show_pred:
        pred_dir = os.path.join(vis.pred_dir, "2024_02_07_08_55_04/pred_train")
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".jpg")])

        if len(pred_files) > 0:
            num_files = len(pred_files)
        else:
            vis.show_pred = False
    
    if vis.show_frustrum:
        frustrum_published = False
        frustrum_path = os.path.join(vis.data_dir, "../others", "frustrum.pt")

    assert num_files > 0, "No files to go through!"

    i = 0

    # Loop through the files
    while not rospy.is_shutdown():
        print(i)

        if rospy.is_shutdown():
            break

        if vis.show_pc1:
            pc_path1 = os.path.join(pc_dir1, pc_files1[i])
            pc1 = torch.load(pc_path1, map_location=torch.device("cpu")).cpu().numpy().astype(np.float32)

            pc1 = vis.correct_z_direction(pc1)
            vis.point_cloud_process1(pc1, publish=True)

        if vis.show_pc2:
            pc_path2 = os.path.join(pc_dir2, pc_files2[i])
            pc2 = torch.load(pc_path2, map_location=torch.device("cpu")).cpu().numpy().astype(np.float32)

            pc2 = vis.correct_z_direction(pc2)
            vis.point_cloud_process2(pc2, publish=True)

        if vis.show_label:
            label_path = os.path.join(label_dir, label_files[i])
            label = torch.load(label_path)
            label += 1
            label = label[np.newaxis, ...].astype(np.uint8) # Shape: (1, H, W)

            # Flip around x axis
            # label = np.flip(label, axis=1)

            vis.occupancy_map_arr(label, RESOLUTION, x=0, y=0)

        if vis.show_pred:
            pred_path = os.path.join(pred_dir, pred_files[i])
            pred = cv2.imread(pred_path)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            # pred = vis.preprocess_image(pred, LOWER_LIM, UPPER_LIM)
            # pred = pred.astype(bool)
            pred = pred[np.newaxis, ...].astype(np.uint8)

            # print(pred)
            vis.grid_map_arr(pred, RESOLUTION, LAYERS, x=0, y=0)
        
        if vis.show_frustrum and not frustrum_published:
            frustrum = torch.load(frustrum_path, map_location=torch.device("cpu")).cpu().numpy().astype(np.float32)
            vis.frustrum_process(frustrum, publish=True)

        if rospy.has_param("dynamic_params_bev/IDX"):
            IDX = rospy.get_param("dynamic_params_bev/IDX")

        if IDX < num_files:
            i = IDX
        else:
            IDX = i

        rospy.sleep(0.2)
