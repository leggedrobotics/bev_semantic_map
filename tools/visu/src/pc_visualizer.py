#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import cv2
import torch

np.set_printoptions(threshold=sys.maxsize)


class PcdVisualizer:
    def __init__(self):
        self.show_pc1 = rospy.get_param("show_pc1")
        self.show_pc2 = rospy.get_param("show_pc2")
        self.data_dir = rospy.get_param("data_dir")
        self.pub_pc1 = rospy.Publisher("point_cloud1", PointCloud2, queue_size=1)
        self.pub_pc2 = rospy.Publisher("point_cloud2", PointCloud2, queue_size=1)

    def point_cloud_process1(self, point_cloud, publish=False):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        # Create a PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_pc1.publish(pointcloud_msg)

    def point_cloud_process2(self, point_cloud, publish=False):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        # Create a PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_pc2.publish(pointcloud_msg)

    def correct_z_direction(self, point_cloud):

        # Increase z value by 0.5
        point_cloud[:, 2] += 0.6

        return point_cloud


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
        pc_dir2 = os.path.join(vis.data_dir, "pc_ext")
        pc_files2 = sorted([f for f in os.listdir(pc_dir2) if f.endswith(".pt")])

        if len(pc_files2) > 0:
            num_files = len(pc_files2)
        else:
            vis.show_pc2 = False

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

            vis.correct_z_direction(pc1)
            vis.point_cloud_process1(pc1, publish=True)

        if vis.show_pc2:
            pc_path2 = os.path.join(pc_dir2, pc_files2[i])
            pc2 = torch.load(pc_path2, map_location=torch.device("cpu")).cpu().numpy().astype(np.float32)

            vis.correct_z_direction(pc2)
            vis.point_cloud_process2(pc2, publish=True)

        if rospy.has_param("dynamic_params_bev/IDX"):
            IDX = rospy.get_param("dynamic_params_bev/IDX")

        if IDX < num_files:
            i = IDX
        else:
            IDX = i

        rospy.sleep(0.2)
