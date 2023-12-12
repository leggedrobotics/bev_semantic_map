#!/usr/bin/env python

"""
Tools for visualizing the data.

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


class DataVisualizer:
    def __init__(self):
        rospy.init_node("data_visualizer", anonymous=False)

        self.pub_pc = rospy.Publisher("/bevnet/pc", PointCloud2, queue_size=1)
        self.pub_map = rospy.Publisher("/bevnet/map", OccupancyGrid, queue_size=1)

    def publish_occ_map(self, arr, res, reference_frame="world", publish=True, x=0, y=0):
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
            self.pub_map.publish(occupancy_grid)

        return occupancy_grid

    def correct_z_direction(self, point_cloud):
        # Increase z value by 0.6
        point_cloud[:, 2] += 0.6

        return point_cloud

    def publish_pc(self, point_cloud, publish=True):
        header = std_msgs.msg.Header()
        header.frame_id = "world"

        # Create a PointCloud2 message
        pointcloud_msg = sensor_msgs.point_cloud2.create_cloud_xyz32(header, point_cloud)

        if publish:
            self.pub_pc.publish(pointcloud_msg)
