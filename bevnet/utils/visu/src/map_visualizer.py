#!/usr/bin/env python

import rospy
from grid_map_msgs.msg import GridMap, GridMapInfo
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import cv2

class NumpyToMapVisualizer:
    def __init__(self, init_node=True):
        rospy.init_node("np_to_gridmap_visualizer", anonymous=False)
        self.pub_grid_map = rospy.Publisher("gridmap", GridMap, queue_size=1)
        self.pub_occupancy_map = rospy.Publisher("occupancy_grid", OccupancyGrid, queue_size=1)

    def gridmap_arr(self, arr, res, layers, reference_frame="map", publish=True, x=0, y=0):
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
            print("publishing")
            self.pub_grid_map.publish(gm_msg)
        return gm_msg

    def occupancy_map_arr(self, arr, res, reference_frame="map", publish=True, x=0, y=0):
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
            print("publishing")
            self.pub_occupancy_map.publish(occupancy_grid)
        return occupancy_grid


if __name__ == "__main__":
    vis = NumpyToMapVisualizer()

    shape = (1, 512, 512)
    arr = np.random.rand(*shape)

    # Threshold arr if bigger than 0.5
    arr[arr > 0.5] = 1
    arr[arr <= 0.5] = 0

    res = 0.1
    layers = ["l1"]

    img = cv2.imread("/home/rschmid/img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[np.newaxis, ...]

    arr = img

    # while not rospy.is_shutdown():
    #     vis.gridmap_arr(arr, res, layers, x=0, y=0)
    #     rospy.sleep(1)

    while not rospy.is_shutdown():
        vis.occupancy_map_arr(arr, res, x=0, y=0)
        rospy.sleep(1)
