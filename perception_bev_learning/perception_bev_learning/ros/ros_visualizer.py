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


if __name__ == "__main__":
    from perception_bev_learning.utils import load_yaml, load_pkl
    from perception_bev_learning import BEV_ROOT_DIR
    import time
    import pickle as pkl
    from os.path import join
    import torch

    vis = SimpleNumpyToRviz()

    with open(join(BEV_ROOT_DIR, "assets", "sample_585.pkl"), "rb") as handle:
        sample = pkl.load(handle)

    (
        imgs,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        target,
        aux,
        img_plots,
        grid_map_resolution,
        pcd_data,
        pcd_sensor_gravity_points,
    ) = sample

    arr = torch.cat([target, aux[0][None], pcd_data[0]], dim=0)

    m = 10
    for i in range(1000):
        vis.gridmap_arr(
            arr.numpy()[:, m:-m, m:-m],
            res=sample[-3],
            layers=["wheel_risk_cvar", "elevation", "pcd_data"],
        )
        vis.pointcloud(pcd_sensor_gravity_points[0].numpy()[:, :3])
        time.sleep(0.1)

    # tf = load_pkl("/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc/processed/crl_rzr_multisense_front_aux_semantic_image_rect_color_compressed/1654644553_662919669.pkl")
    # gridmap = load_pkl("/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc/processed/crl_rzr_traversability_map_map_micro/1654644555_898578510.pkl")

    # pointcloud = load_pkl("/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc/processed/crl_rzr_traversability_map_map_micro/1654644555_898578510.pkl")

    # for i in range(100):
    #     vis.gridmap(gridmap)
    #     vis.tf(tf)
    #     time.sleep(0.1)
