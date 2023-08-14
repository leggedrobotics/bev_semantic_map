#!/usr/bin/env python

"""
Saves geometric features on images. Saves features from normal cloud and principal cloud.
Uses fixed tf from realsense to camera.

Author: Robin Schmid
Date: Nov 2022
"""

import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T

import rospy
import rosbag
import tf2_ros
import torch
import subprocess
import yaml


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
torch.set_printoptions(edgeitems=100)

SINGLE_IMG = True
VISUALIZE = True


def multiply_quaternion(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                     x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                     -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                     x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])


# rosbag_path = "/home/rschmid/pointcloud_bags/hoengg_pc.bag"
rosbag_path = "/home/rschmid/pointcloud_bags/hoengg_pc_vel.bag"
# rosbag_path = "/home/rschmid/RosBags/sa_geom_train/sa_walk_pc.bag"

# South Africa
# TF_TRANS_REALSENSE_TO_CAMERA = [-0.014204385537950434, 0.2756737895147698, 0.08061656990524951]
# TF_ROT_REALSENSE_TO_CAMERA = [0.24981959957596891, -0.011532330452940665, -0.007644459970728101, -0.9681935422495768]

# South Africa
# TF_TRANS_VELODYNE_TO_CAMERA = [0.0021375334062859854, -0.03956706739849679, 0.0920467808]
# TF_ROT_VELODYNE_TO_CAMERA = [0.0087744421888676, -0.7068219147571273, 0.7073014361400815, -0.007104112718762427]

# Perugia
TF_TRANS_VELODYNE_TO_CAMERA = [0.0036920364377833652, -0.04315291620837264, 0.08874986509999999]
TF_ROT_VELODYNE_TO_CAMERA = [0.009399369655087067, -0.7057808978358807, 0.7083022231566545, -0.009640371953725439]

# Perugia
# TF_TRANS_BPEARL_TO_CAMERA = [-0.16591687972787286, -0.160695551724784, -0.022342083791627387]
# TF_ROT_BPEARL_TO_CAMERA = [0.0035049818099763264, 0.013120576338846446, 0.382490692842936, -0.923859532324898]
# # Correction for bpearl orientation, rotate around x-axis by 90 degrees
# q1 = [0.0035049818099763264, 0.013120576338846446, 0.382490692842936, -0.923859532324898]
# q2 = [0.70710678118, 0.70710678118, 0, 0]
# TF_ROT_BPEARL_TO_CAMERA = multiply_quaternion(q1, q2)

start_time_offset = 0
stop_time_offset = 0


def get_bag_info(rosbag_path: str) -> dict:
    # This queries rosbag info using subprocess and get the YAML output to parse the topics
    info_dict = yaml.safe_load(
        subprocess.Popen(["rosbag", "info", "--yaml", rosbag_path], stdout=subprocess.PIPE).communicate()[0]
    )
    return info_dict


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

        self.image_topic_cam4 = "/alphasense_driver_ros/cam4/debayered"

        self.point_cloud_topic_vel = "/point_cloud_out"
        # /pointcloud_transformer/output_pcl2, /point_cloud_filter/lidar/point_cloud_filtered, /depth_camera_rear/depth/color/points_filtered

        # Set up tf buffer, broadcaster, listener
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self.listener = tf.TransformListener()
        self._cv_bridge = CvBridge()

        # Camera 4 (front), South Africa
        # self.camera_info_cam4 = CameraInfo()
        # self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        # self.camera_info_cam4.height = 1080
        # self.camera_info_cam4.width = 1440
        # self.camera_info_cam4.distortion_model = "equidistant"
        # self.camera_info_cam4.D = [-0.0480706813, 0.0129997684, -0.0112199955, 0.0026955514]
        # self.camera_info_cam4.K = [699.2284099702, 0.0, 711.8009584441, 0.0, 698.546880367, 524.7993478318, 0.0, 0.0, 1.0]
        # self.camera_info_cam4.P = [699.2284099702, 0.0, 711.8009584441, 0.0, 0.0, 698.546880367, 524.7993478318, 0.0 ,0.0, 0.0, 1.0, 0.0]
        # self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # self.camera_model_cam4 = PinholeCameraModel()
        # self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # Camera 4 (existing in repo), Perugian (best)
        self.camera_info_cam4 = CameraInfo()
        self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        self.camera_info_cam4.height = 540
        self.camera_info_cam4.width = 720
        self.camera_info_cam4.distortion_model = "plumb_bob"
        self.camera_info_cam4.K = [347.548139773951, 0.0, 342.454373227748, 0.0, 347.434712422309, 271.368057185649, 0.0, 0.0, 1.0]
        self.camera_info_cam4.P = [347.548139773951, 0.0, 342.454373227748, 0.0, 0.0, 347.434712422309, 271.368057185649, 0.0,
                      0.0, 0.0, 1.0, 0.0]
        self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_model_cam4 = PinholeCameraModel()
        self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # self.camera_info_cam4 = CameraInfo()
        # self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        # self.camera_info_cam4.height = 540
        # self.camera_info_cam4.width = 720
        # self.camera_info_cam4.distortion_model = "plumb_bob"
        # self.camera_info_cam4.K = [407.548139773951, 0.0, 402.454373227748, 0.0, 407.434712422309, 271.368057185649, 0.0, 0.0, 1.0]
        # self.camera_info_cam4.P = [407.548139773951, 0.0, 402.454373227748, 0.0, 0.0, 407.434712422309, 271.368057185649, 0.0,
        #               0.0, 0.0, 1.0, 0.0]
        # self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # self.camera_model_cam4 = PinholeCameraModel()
        # self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # Perugia 2 (from Matias)
        # self.camera_info_cam4 = CameraInfo()
        # self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        # self.camera_info_cam4.height = 540
        # self.camera_info_cam4.width = 720
        # self.camera_info_cam4.distortion_model = "equidistant"
        # self.camera_info_cam4.D = [-0.0407923963, 0.0001333626, -0.0017346622, 0.0002102967]
        # self.camera_info_cam4.K = [347.8630203101, 0.0, 362.3369694313, 0.0, 347.9862908151, 269.92042226, 0.0, 0.0, 1.0]
        # self.camera_info_cam4.P = [347.8630203101, 0.0, 362.3369694313, 0.0, 0.0, 347.9862908151, 269.92042226, 0.0 ,0.0, 0.0, 1.0, 0.0]
        # self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # self.camera_model_cam4 = PinholeCameraModel()
        # self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # Perugia 3 (old one from Jonas)
        # self.camera_info_cam4 = CameraInfo()
        # self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"  # Important, use helper frame for projection
        # self.camera_info_cam4.height = 540
        # self.camera_info_cam4.width = 720
        # self.camera_info_cam4.distortion_model = "equidistant"
        # self.camera_info_cam4.D = [-0.0435067508, 0.0048952632, -0.0035388936, 0.0003353017]
        # self.camera_info_cam4.K = [349.9204665671, 0.0, 344.3032414701, 0.0, 349.8220154122, 258.449836991, 0.0, 0.0, 1.0]
        # self.camera_info_cam4.P = [349.9204665671, 0.0, 344.3032414701, 0.0, 0.0, 349.8220154122, 258.449836991, 0.0, 0.0, 0.0, 1.0, 0.0]
        # self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # self.camera_model_cam4 = PinholeCameraModel()
        # self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        # Höngg
        # self.camera_info_cam4 = CameraInfo()
        # self.camera_info_cam4.header.frame_id = "cam4_sensor_frame_helper"
        # self.camera_info_cam4.height = 540
        # self.camera_info_cam4.width = 720
        # self.camera_info_cam4.distortion_model = "equidistant"
        # self.camera_info_cam4.K = [349.5636550689, 0.0, 357.2746308879, 0.0, 349.4046775293, 264.3108985411, 0.0, 0.0, 1.0]
        # self.camera_info_cam4.P = [349.5636550689, 0.0, 357.2746308879, 0.0, 0.0, 349.4046775293, 264.3108985411, 0.0, 0.0, 0.0, 1.0, 0.0]
        # self.camera_info_cam4.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # self.camera_model_cam4 = PinholeCameraModel()
        # self.camera_model_cam4.fromCameraInfo(self.camera_info_cam4)

        self.image = None
        self.image_topic = self.image_topic_cam4
        self.camera_model = self.camera_model_cam4
        self.camera_info = self.camera_info_cam4
        self.point_cloud_vel = None
        self.point_cloud_real = None

        # Set up subscribers
        # self._point_cloud_sub_vel = rospy.Subscriber(self.point_cloud_topic_vel, PointCloud2, self.point_cloud_cb,
        #                                              queue_size=1)
        # self._image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=1)
        # self._image_cache = message_filters.Cache(self._image_sub, cache_size=20)

        self.count = 0

        # create_dir(f"../../samples/points/cam3")
        # create_dir(f"../../samples/points")
        # create_dir(f"../../samples/points/cam5")

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

        for data in point_cloud2.read_points(ros_cloud, skip_nans=False):
            data_p = data[:3]
            data_p = np.matmul(R[:3, :3], np.array(data_p)) + position
            points_list.append(tuple(data_p) + data[3:])
        return np.array(points_list)

    def project_on_image(self, principal_cloud, normal_cloud, time_stamp):
        # Create a blank image
        blank_image = np.zeros((self.camera_info_cam4.height, self.camera_info_cam4.width, 3), np.uint8)

        # Tf realsense to cam4 for sa_walk dataset
        # trans = TF_TRANS_REALSENSE_TO_CAMERA
        # rot = TF_ROT_REALSENSE_TO_CAMERA

        trans = TF_TRANS_VELODYNE_TO_CAMERA
        rot = TF_ROT_VELODYNE_TO_CAMERA

        # trans = TF_TRANS_BPEARL_TO_CAMERA
        # rot = TF_ROT_BPEARL_TO_CAMERA

        pose_normal = [*trans, *rot]
        normal_points = self.rospcmsg_to_pcarray(normal_cloud, pose_normal)

        # No shift for principal cloud needed
        pose_principal = [0, 0, 0, 0, 0, 0, 1]
        principal_points = self.rospcmsg_to_pcarray(principal_cloud, pose_principal)

        # Check if number of points in normal and principal cloud are different, if yes remove the extra points
        if len(normal_points) != len(principal_points):
            rospy.logwarn("Principal and normal point clouds have different number of points!")
            rospy.logwarn("Num normal points: %d, num principal points: %d", len(normal_points), len(principal_points))
            if len(normal_points) > len(principal_points):
                normal_points = normal_points[:len(principal_points)]
            elif len(normal_points) < len(principal_points):
                principal_points = principal_points[:len(normal_points)]

        # Index of normal points which contain NaNs
        # Remove elements at column index where either normal_points or principal_points contains NaNs
        nan_idx_normal = np.argwhere(np.isnan(normal_points))[:, 0]
        nan_idx_principal = np.argwhere(np.isnan(principal_points))[:, 0]
        nan_idx = np.unique(np.concatenate((nan_idx_normal, nan_idx_principal)))
        normal_points = np.delete(normal_points, nan_idx[:len(normal_points)], axis=0)
        principal_points = np.delete(principal_points, nan_idx, axis=0)
        # print(len(normal_points))
        # print(len(principal_points))

        # Filter out points behind camera
        i_points_outside_z_limit = np.where(normal_points[:, 2] <= 0)[0]

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

        i_points_outside_left = np.where(np.abs(np.sign(np.dot(normal_points[:, :3], left)) + inside_left_sign) <= 0)[0]
        i_points_outside_right = np.where(np.abs(np.sign(np.dot(normal_points[:, :3], right)) + inside_right_sign) <= 0)[0]
        i_points_outside_top = np.where(np.abs(np.sign(np.dot(normal_points[:, :3], top)) + inside_top_sign) <= 0)[0]
        i_points_outside_bottom = np.where(np.abs(np.sign(np.dot(normal_points[:, :3], bottom)) + inside_bottom_sign) <= 0)[0]

        i_outlier_points = np.unique(np.hstack([i_points_outside_left,
                                                i_points_outside_right,
                                                i_points_outside_top,
                                                i_points_outside_bottom,
                                                i_points_outside_z_limit]))
        inlier_mask = np.ones(len(normal_points), dtype=bool)
        inlier_mask[i_outlier_points] = False

        normal_points = normal_points[inlier_mask]
        principal_points = principal_points[inlier_mask]

        # Project points to camera frame
        points_ = np.hstack([normal_points[:, :3], np.ones((normal_points[:, :3].shape[0], 1))])
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
        fov_pixels = np.round(pixels[bool_fov_pixels]).astype(int)

        normal_points = normal_points[bool_fov_pixels]
        # Remove column 6 of normal_points, this contains large values, represents RGB value of PC
        normal_points = np.delete(normal_points, 6, 1)
        # for i in range(len(normal_points)):
        #     print("length:", np.sqrt(normal_points[i, 0]**2 + normal_points[i, 1]**2 + normal_points[i, 2]**2))

        principal_points = principal_points[bool_fov_pixels]
        features = np.hstack([normal_points[:, 3:], principal_points])

        # Create empty image and fill pixel locations with features
        empty_img = torch.empty((size[1], size[0], features.shape[1]), dtype=torch.float32)
        empty_img[:] = np.nan
        empty_img[fov_pixels[:, 1], fov_pixels[:, 0]] = torch.from_numpy(features).type(torch.float32)

        # Resize image with features to 448x448
        empty_img = empty_img.permute(2, 0, 1)

        # empty_img = T.Resize((448, 448), T.InterpolationMode.NEAREST)(empty_img)

        # Also crop on the sides!!!
        empty_img = T.Compose([T.Resize(448, T.InterpolationMode.NEAREST), T.CenterCrop(448)])(empty_img)

        empty_img = empty_img.permute(1, 2, 0)
        empty_img = empty_img.numpy()

        # print(np.count_nonzero(~np.isnan(empty_img[:,:,0])))

        # Visualize the normal angle for debugging
        # xy = empty_img[:, :, :2]
        # z = empty_img[:, :, 2]
        # xy_norm = np.linalg.norm(xy, axis=2)
        # vis_img1 = np.arctan(xy_norm / z)
        # vis_img2 = np.linalg.norm(empty_img[:, :, :3], axis=2)
        # vis_img = np.hstack((vis_img1, vis_img2))
        # cv2.imshow("vis_img", vis_img)
        # cv2.waitKey(0)

        np.save(f"../../../samples/features/{time_stamp}.npy", empty_img)
        np.save(f"../../../samples/fov_pixels/{time_stamp}.npy", fov_pixels)

        if VISUALIZE:
            image_with_points = blank_image.copy()
            # print("Num pixels:", len(fov_pixels))
            for i, pixel in enumerate(fov_pixels):
                cv2.circle(image_with_points, (np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)),
                           radius=3, color=(0, 0, 255))  # Color = red

            res = cv2.addWeighted(blank_image.copy(), 0.3, image_with_points, 0.7, 0.0)
            # cv2.imshow("res", res)
            # cv2.waitKey(0)

            cv2.imwrite(f"../../../samples/points/{time_stamp}.png", res)

        self.count += 1
        # print(self.count)

    def run(self):
        with rosbag.Bag(rosbag_path, "r") as bag:
            start_time = rospy.Time.from_sec(bag.get_start_time() + start_time_offset)
            end_time = rospy.Time.from_sec(bag.get_end_time() - stop_time_offset)

            valid_topics = ["/pointcloud_extractor/normal_cloud", "/principal_extractor/principal_cloud"]
            rosbag_info_dict = get_bag_info(rosbag_path)
            total_msgs = sum([x["messages"] for x in rosbag_info_dict["topics"] if x["topic"] in valid_topics])

            with tqdm(
                    total=len(valid_topics)*total_msgs,
                    desc="Progress",
                    colour="green",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:50}{r_bar}",
            ) as pbar:
                for (topic, msg, ts) in bag.read_messages(topics=None, start_time=start_time, end_time=end_time):
                    if topic == "/pointcloud_extractor/principal_cloud":
                        # print("Received principal cloud at", ts)
                        principal_cloud = msg
                    elif topic == "/pointcloud_extractor/normal_cloud":
                        # print("Received normal cloud at", ts)
                        normal_cloud = msg
                    # Check manually if need to add at pbar.n even or odd - check warnings if not same number of points
                    # if (pbar.n % 2) == 0 and pbar.n != 0:
                    if (pbar.n % 2) == 1 and pbar.n != 0:
                        self.project_on_image(principal_cloud, normal_cloud, ts)
                    pbar.update(1)
                    # For debugging
                    # if pbar.update(1) >= 3:
                    #     break


if __name__ == "__main__":
    print("Start")

    # Initialize and run node
    projector = PointCloudProjector()
    projector.run()
