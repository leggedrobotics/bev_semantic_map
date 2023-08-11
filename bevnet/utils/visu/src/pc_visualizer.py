#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import torch

def visualize_pointcloud():
    rospy.init_node('pointcloud_visualizer', anonymous=True)
    pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)

    rate = rospy.Rate(10)  # Publish rate in Hz

    points_in_base_frame = (torch.rand((5000, 3)) - 0.5) * 20

    while not rospy.is_shutdown():
        # Generate a random 3D point cloud (replace with your actual point cloud data)

        # Convert the Torch tensor to a numpy array
        points_np = points_in_base_frame.numpy().astype(np.float32)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"  # Change to your desired frame ID

        # Create a PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, points_np)

        # Publish the PointCloud2 message
        pub.publish(pointcloud_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        visualize_pointcloud()
    except rospy.ROSInterruptException:
        pass

