#!/usr/bin/env python

"""
Project both realsense and velodyne pointclouds on the image plane.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import cv2
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import message_filters

try:
    import ros_numpy
except Exception:
    pass

from image_geometry import PinholeCameraModel
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import tf
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix

np.set_printoptions(threshold=sys.maxsize)

NUM_IMAGES = 200
SINGLE_IMG = True


def create_dir(dir_name):
    """
    Creates a directory for a given name if it does not exist already
    """
    pkg_path = os.path.join(os.path.dirname(os.path.join(os.getcwd(), __file__)))
    if not os.path.exists(os.path.join(pkg_path, dir_name)):
        print("Creating %s directory" % dir_name)
        os.mkdir(os.path.join(pkg_path, dir_name))


class PointCloudProjector(object):
    def __init__(self):
        rospy.init_node("projector")

        self.image_topic_cam3 = "/alphasense_driver_ros/cam3/debayered"
        self.image_topic_cam4 = "/alphasense_driver_ros/cam4/debayered"
        self.image_topic_cam5 = "/alphasense_driver_ros/cam5/debayered"

        self.point_cloud_topic_vel = "/point_cloud_filter/lidar/point_cloud_filtered"
        self.point_cloud_topic_real = "/depth_camera_rear/depth/color/points_filtered"
        # /pointcloud_transformer/output_pcl2, /point_cloud_filter/lidar/point_cloud_filtered, /depth_camera_rear/depth/color/points_filtered

        # Set up tf buffer, broadcaster, listener
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self.listener = tf.TransformListener()
        self._cv_bridge = CvBridge()

        # Camera 3 (left)
        self.camera_info_cam3 = CameraInfo()
        self.camera_info_cam3.header.frame_id = "cam3_sensor_frame_helper"  # Important, use helper frame for projection
        self.camera_info_cam3.height = 1080
        self.camera_info_cam3.width = 1440
        self.camera_info_cam3.distortion_model = "equidistant"
        self.camera_info_cam3.D = [-0.0391589409, 0.0025508685, -0.0070315976, 0.0028446106]
        self.camera_info_cam3.K = [699.6648558052, 0.0, 684.6546232939, 0.0, 698.9976002222, 518.1907596555, 0.0, 0.0, 1.0]
        self.camera_info_cam3.P = [699.6648558052, 0.0, 684.6546232939, 0.0, 0.0, 698.9976002222, 518.1907596555, 0.0 ,0.0, 0.0, 1.0, 0.0]
        self.camera_info_cam3.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_model_cam3 = PinholeCameraModel()
        self.camera_model_cam3.fromCameraInfo(self.camera_info_cam3)

        # Camera 4 (front)
        self.camera_info_cam4 = CameraInfo()
        self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        self.camera_info_cam4.height = 1080
        self.camera_info_cam4.width = 1440
        self.camera_info_cam4.distortion_model = "equidistant"
        self.camera_info_cam4.D = [-0.0480706813, 0.0129997684, -0.0112199955, 0.0026955514]
        self.camera_info_cam4.K = [699.2284099702, 0.0, 711.8009584441, 0.0, 698.546880367, 524.7993478318, 0.0, 0.0, 1.0]
        self.camera_info_cam4.P = [699.2284099702, 0.0, 711.8009584441, 0.0, 0.0, 698.546880367, 524.7993478318, 0.0 ,0.0, 0.0, 1.0, 0.0]
        self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_model_cam4 = PinholeCameraModel()
        self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # Camera 5 (right)
        self.camera_info_cam5 = CameraInfo()
        self.camera_info_cam5.header.frame_id = "cam5_sensor_frame_helper"  # Important, use helper frame for projection
        self.camera_info_cam5.height = 1080
        self.camera_info_cam5.width = 1440
        self.camera_info_cam5.distortion_model = "equidistant"
        self.camera_info_cam5.D = [-0.0393569867, -0.0015711557, -0.0003396351, -0.0001519304]
        self.camera_info_cam5.K = [700.1349034076, 0.0, 761.3978917011, 0.0, 699.958743393, 552.7799875257, 0.0, 0.0, 1.0]
        self.camera_info_cam5.P = [700.1349034076, 0.0, 761.3978917011, 0.0, 0.0, 699.958743393, 552.7799875257, 0.0 ,0.0, 0.0, 1.0, 0.0]
        self.camera_info_cam5.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_model_cam5 = PinholeCameraModel()
        self.camera_model_cam5.fromCameraInfo(self.camera_info_cam5)

        # Camera 6 (back)
        self.camera_info_cam6 = CameraInfo()
        self.camera_info_cam6.header.frame_id = "cam6_sensor_frame_helper"  # Important, use helper frame for projection
        self.camera_info_cam6.height = 1080
        self.camera_info_cam6.width = 1440
        self.camera_info_cam6.distortion_model = "equidistant"
        self.camera_info_cam6.D = [-0.0461433203, 0.0125454629, -0.0102876501, 0.0022414551]
        self.camera_info_cam6.K = [695.0792829233, 0.0, 687.1167332592, 0.0, 694.3314443904, 522.4848030013, 0.0, 0.0, 1.0]
        self.camera_info_cam6.P = [695.0792829233, 0.0, 687.1167332592, 0.0, 0.0, 694.3314443904, 522.4848030013, 0.0, 0.0,
                              0.0, 1.0, 0.0]
        self.camera_info_cam6.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_model_cam6 = PinholeCameraModel()
        self.camera_model_cam6.fromCameraInfo(self.camera_info_cam6)

        self.image = None
        self.image_topic = self.image_topic_cam4
        self.camera_model = self.camera_model_cam4
        self.camera_info = self.camera_info_cam4
        self.point_cloud_vel = None
        self.point_cloud_real = None

        # Set up subscribers
        self._point_cloud_sub_vel = rospy.Subscriber(self.point_cloud_topic_vel, PointCloud2, self.point_cloud_cb,
                                                     queue_size=1)
        self._point_cloud_sub_real = message_filters.Subscriber(self.point_cloud_topic_real, PointCloud2, queue_size=1)
        self._point_cloud_cache_real = message_filters.Cache(self._point_cloud_sub_real, cache_size=20)
        self._image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=1)
        self._image_cache = message_filters.Cache(self._image_sub, cache_size=20)
        # self._image_sub2 = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

        self.count = 0

        create_dir(f"../../../samples/points/cam3")
        create_dir(f"../../../samples/points/cam4")
        create_dir(f"../../../samples/points/cam5")

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        rospy.loginfo("Shutdown")
        return

    def convert_ros_msg_to_cv2(self, ros_data, image_encoding='bgr8'):
        try:
            return self._cv_bridge.imgmsg_to_cv2(ros_data, image_encoding)
        except CvBridgeError as e:
            raise e

    def convert_ros_compressed_to_cv2(self, compressed_msg):
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def convert_ros_compressed_msg_to_ros_msg(self, compressed_msg,
                                              encoding='bgr8'):
        cv2_img = self.convert_ros_compressed_to_cv2(compressed_msg)
        ros_img = self._cv_bridge.cv2_to_imgmsg(cv2_img, encoding=encoding)
        ros_img.header = compressed_msg.header
        return ros_img

    def rospcmsg_to_pcarray(self, ros_cloud, cam_pose):
        position = np.array(cam_pose[:3])
        R = np.array(quaternion_matrix(cam_pose[3:]))

        points_list = []

        for data in point_cloud2.read_points(ros_cloud, skip_nans=True):
            data_p = data[:3]
            data_p = np.matmul(R[:3, :3], np.array(data_p)) + position
            points_list.append(tuple(data_p) + data[3:])
        return np.array(points_list)

    def point_cloud_cb(self, msg):
        rospy.loginfo(f"[image_cb]: Point cloud received")

        self.point_cloud_vel = msg

        if self.capture:
            self.capture_images()

    def capture_images(self):

        if self.point_cloud_vel is None:
            rospy.logwarn("[capture_image]: Point cloud velodyne not received")
            return

        # Get latest image
        self.image = self._image_cache.getElemBeforeTime(self.point_cloud_vel.header.stamp)
        self.point_cloud_real = self._point_cloud_cache_real.getElemBeforeTime(self.point_cloud_vel.header.stamp)

        if self.image is None:
            rospy.logwarn("[capture_image]: Image not received")
            return
        if self.point_cloud_real is None:
            rospy.logwarn("[capture_image]: Point cloud realsense not received")
            return

        # Add point from velodyne point cloud
        try:
            self.listener.waitForTransformFull(self.camera_info.header.frame_id, self.image.header.stamp,
                                               self.point_cloud_real.header.frame_id, self.point_cloud_vel.header.stamp,
                                               "odom", rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransformFull(self.camera_info.header.frame_id, self.image.header.stamp,
                                                             self.point_cloud_real.header.frame_id,
                                                             self.point_cloud_real.header.stamp, "odom")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logwarn("[capture_image]: Transform not found")
            return

        pose_vel = [*trans, *rot]
        points_vel = self.rospcmsg_to_pcarray(self.point_cloud_vel, pose_vel)[:, :3]

        try:
            self.listener.waitForTransformFull(self.camera_info.header.frame_id, self.image.header.stamp,
                                               self.point_cloud_real.header.frame_id, self.point_cloud_vel.header.stamp,
                                               "odom", rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransformFull(self.camera_info.header.frame_id, self.image.header.stamp,
                                                             self.point_cloud_real.header.frame_id,
                                                             self.point_cloud_real.header.stamp, "odom")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logwarn("[capture_image]: Transform not found")
            return

        pose_real = [*trans, *rot]
        points_real = self.rospcmsg_to_pcarray(self.point_cloud_real, pose_real)[:, :3]
        points = np.concatenate((points_real, points_vel), axis=0)

        # Convert image msg to cv
        input_image = self._cv_bridge.imgmsg_to_cv2(self.image, desired_encoding='passthrough')

        # Filter out points behind camera
        i_points_outside_z_limit = np.where(points[:, 2] <= 0)[0]

        # Get camera image size
        size = self.camera_model.fullResolution()

        # Filter out points outside of camera frustum
        bottom_left = np.asarray(self.camera_model.projectPixelTo3dRay([0, 0]))
        bottom_right = np.asarray(
            self.camera_model.projectPixelTo3dRay([size[0] - 1, 0])
        )
        top_left = np.asarray(self.camera_model.projectPixelTo3dRay([0, size[1] - 1]))
        top_right = np.asarray(
            self.camera_model.projectPixelTo3dRay([size[0] - 1, size[1] - 1])
        )
        centroid = np.mean(np.vstack([bottom_left, bottom_right, top_left, top_right]), 0)
        left = np.cross(bottom_left, top_left)
        right = np.cross(bottom_right, top_right)
        top = np.cross(top_left, top_right)
        bottom = np.cross(bottom_left, bottom_right)

        inside_left_sign = np.sign(np.dot(left, centroid))
        inside_right_sign = np.sign(np.dot(right, centroid))
        inside_top_sign = np.sign(np.dot(top, centroid))
        inside_bottom_sign = np.sign(np.dot(bottom, centroid))

        i_points_outside_left = np.where(np.abs(np.sign(np.dot(points, left)) + inside_left_sign) <= 0)[0]
        i_points_outside_right = np.where(np.abs(np.sign(np.dot(points, right)) + inside_right_sign) <= 0)[0]
        i_points_outside_top = np.where(np.abs(np.sign(np.dot(points, top)) + inside_top_sign) <= 0)[0]
        i_points_outside_bottom = np.where(np.abs(np.sign(np.dot(points, bottom)) + inside_bottom_sign) <= 0)[0]

        i_outlier_points = np.unique(np.hstack([i_points_outside_left,
                                                i_points_outside_right,
                                                i_points_outside_top,
                                                i_points_outside_bottom,
                                                i_points_outside_z_limit]))
        inlier_mask = np.ones(len(points), dtype=bool)
        inlier_mask[i_outlier_points] = False
        points = points[inlier_mask]

        # Project points to camera frame
        points_ = np.hstack([points, np.ones((points.shape[0], 1))])
        xyw = np.matmul(self.camera_model.P, points_.transpose())
        u = xyw[0] / xyw[2]
        v = xyw[1] / xyw[2]
        pixels = np.asarray(np.vstack([u, v]).transpose())

        # Filter out points outside of field of view
        size = self.camera_model.fullResolution()
        bool_fov_pixels = np.logical_and.reduce(
            [
                pixels[:, 0] >= 0,
                pixels[:, 0] < size[0],
                pixels[:, 1] >= 0,
                pixels[:, 1] < size[1],
            ]
        )

        # Draw pixels in field of view on image
        fov_pixels = pixels[bool_fov_pixels]

        image_with_points = input_image.copy()
        # print("Num pixels:", len(fov_pixels))
        for i, pixel in enumerate(fov_pixels):
            cv2.circle(image_with_points, (np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)),
                       radius=2, color=(0, 0, 255))  # Color = red

        cv2.imwrite(f"points/cam4/{self.image.header.stamp}.png", image_with_points)

        self.count += 1
        print(self.count)

        # Change flag after capturing image
        self.capture = False

    def run(self):
        for _ in enumerate(range(NUM_IMAGES)):

            self.capture = True
            while self.capture:
                pass


if __name__ == "__main__":
    print("Start")

    # Initialize and run node
    projector = PointCloudProjector()
    projector.run()
