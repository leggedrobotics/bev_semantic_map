import sys
import os
from pathlib import Path
import argparse
import rospy
import ros_numpy
import numpy as np
from tqdm import tqdm
import rosbag
import pickle as pkl
from os.path import join
from cv_bridge import CvBridge
import cv2
import yaml
import subprocess

from perception_bev_learning.preprocessing import BagTfTransformerWrapper, get_bag_info
from perception_bev_learning.preprocessing import fix_rosbags, merge_bags_single
from perception_bev_learning.preprocessing import (
    PDC_DATATYPE,
    COUNTER_DIGITS,
    SECONDS_DIGITS,
    NSECONDS_DIGITS,
    IMAGE_OUTPUT_FORMAT,
)
from perception_bev_learning.preprocessing import (
    counter_to_str,
    secs_to_str,
    nsecs_to_str,
)

import warnings


def convert_gridmap_uint8(msg):
    res = []
    for j, data in enumerate(msg.data):
        dims = tuple(map(lambda x: x.size, data.layout.dim))
        buf = np.frombuffer(data.data, np.dtype(np.uint8)).reshape(dims)
        res.append(buf)
    return (
        {
            "data": np.stack(res),
            "layers": msg.layers,
            "basic_layers": msg.basic_layers,
            "scaling": (msg.scaling_min, msg.scaling_max),
            "resolution": msg.info.resolution,
            "length": (msg.info.length_x, msg.info.length_y),
            "position": (
                msg.info.pose.position.x,
                msg.info.pose.position.y,
                msg.info.pose.position.z,
            ),
            "orientation_xyzw": (
                msg.info.pose.orientation.x,
                msg.info.pose.orientation.y,
                msg.info.pose.orientation.z,
                msg.info.pose.orientation.w,
            ),
        },
    )


def convert_gridmap_float32(msg):
    res = []
    layers_out = []
    for layer in [
        "elevation_raw",
        "num_points",
        "reliable",
        "elevation",
        "unknown",
        "wheel_risk_cvar",
        "costmap_max_velocity",
    ]:
        if layer in msg.layers:
            # extract grid_map layer as numpy array
            layers_out.append(layer)
            data_list = msg.data[msg.layers.index(layer)].data
            layout_info = msg.data[msg.layers.index(layer)].layout
            assert layout_info.data_offset == 0
            assert layout_info.dim[1].stride == layout_info.dim[1].size
            assert layout_info.dim[0].label == "column_index"
            n_cols = layout_info.dim[0].size
            assert layout_info.dim[1].label == "row_index"
            n_rows = layout_info.dim[1].size
            data_in_layer = np.reshape(np.array(data_list), (n_rows, n_cols))
            data_in_layer = data_in_layer[::-1, ::-1].transpose().astype(np.float32)
            res.append(data_in_layer)

    return (
        {
            "data": np.stack(res),
            "layers": layers_out,
            "basic_layers": msg.basic_layers,
            "resolution": msg.info.resolution,
            "length": (msg.info.length_x, msg.info.length_y),
            "position": (
                msg.info.pose.position.x,
                msg.info.pose.position.y,
                msg.info.pose.position.z,
            ),
            "orientation_xyzw": (
                msg.info.pose.orientation.x,
                msg.info.pose.orientation.y,
                msg.info.pose.orientation.z,
                msg.info.pose.orientation.w,
            ),
        },
    )


def get_tf(self, header, frame, bag_transformer):
    try:
        trans, rot = bag_transformer.lookupTransform(
            frame, header.frame_id, header.stamp
        )
        return trans, rot
    except:
        print("failed lookup")
        return None, None


class RosbagConverter:
    def __init__(self, directory):
        fix_rosbags(directory)
        tf_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find("_bev_tf_") != -1
        ]
        trav_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find("_bev_trav_") != -1
        ]
        image_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find("_bev_color_") != -1
        ]
        pointcloud_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find("_bev_velodyne_") != -1
        ]

        output_bag_tf = os.path.join(directory, "merged_tf.bag")

        # merge tf bags
        if not os.path.exists(output_bag_tf):
            for p in tf_rosbags:
                subprocess.run(["rosbag", "reindex", p])
                subprocess.run(["rosbag", "decompress", p])

            total_included_count, total_skipped_count = merge_bags_single(
                input_bag=tf_rosbags,
                output_bag=output_bag_tf,
                topics="/tf /tf_static",
                verbose=True,
            )

        self.output_folder = os.path.join(directory, "processed")
        self.tf_listener = BagTfTransformerWrapper(output_bag_tf)
        self.reference_frame = "crl_rzr/map"
        self.bridge = CvBridge()

        extraction_tasks = {
            "trav": trav_rosbags,
            #    "pointcloud": pointcloud_rosbags,
            #    "image": image_rosbags
        }
        self.convert_rosbags(extraction_tasks)

    def store_camera_info(self, topic, msg):
        di = {}
        camera_info = {
            method_name: getattr(msg, method_name)
            for method_name in dir(type(msg))
            if not callable(getattr(type(msg), method_name))
            and method_name[0] != "_"
            and method_name.find("roi") == -1
            and method_name.find("header") == -1
        }
        for k, v in camera_info.items():
            if type(v) is tuple:
                camera_info[k] = list(v)
        filename = join(
            self.output_folder,
            topic[1:].replace("/", "_").replace("camera_info", "compressed"),
            "camera_info.yaml",
        )
        try:
            with open(filename, "w") as f:
                yaml.dump(camera_info, f)
        except:
            pass

    def store_compressed_image(self, topic, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )
        p = join(self.output_folder, topic[1:].replace("/", "_"))
        filename = join(
            p,
            f"{secs_to_str(msg.header.stamp.secs)}_{nsecs_to_str(msg.header.stamp.nsecs)}{IMAGE_OUTPUT_FORMAT}",
        )

        suc = self.store(topic, msg, None, msg.header)
        suc *= self.store(
            topic,
            msg,
            None,
            msg.header,
            tag="_map__base_link_tf",
            tar_frame="crl_rzr/base_link",
        )
        suc *= self.store(
            topic,
            msg,
            None,
            msg.header,
            tag="_map__sensor_origin_link_tf",
            tar_frame="crl_rzr/sensor_origin_link",
        )
        suc *= self.store(
            topic, msg, None, msg.header, tag="_map__odom_tf", tar_frame="crl_rzr/odom"
        )
        suc *= self.store(
            topic,
            msg,
            None,
            msg.header,
            tag="_map__aux_camera_optical_frame_tf",
            tar_frame=msg.header.frame_id.replace(
                "aux_camera_frame", "aux_camera_optical_frame"
            ),
        )
        suc *= self.store(
            topic,
            msg,
            None,
            msg.header,
            tag="_map__left_camera_optical_frame_tf",
            tar_frame=msg.header.frame_id.replace(
                "aux_camera_frame", "left_camera_optical_frame"
            ),
        )
        suc *= self.store(
            topic,
            msg,
            None,
            msg.header,
            tag="_map__right_camera_optical_frame_tf",
            tar_frame=msg.header.frame_id.replace(
                "aux_camera_frame", "right_camera_optical_frame"
            ),
        )

        # only write if tf exists
        if suc:
            cv2.imwrite(filename, cv_img)

    def store_gridmap_float32(self, topic, msg):
        res = convert_gridmap_float32(msg)
        self.store(topic, msg, res, msg.info.header)

    def store_gridmap_uint8(self, topic, msg):
        res = convert_gridmap_uint8(topic, msg)
        self.store(topic, msg, res, msg.info.header)

    def store_pointcloud(self, topic, msg):
        output = {}
        total_bytes = (
            msg.fields[-1].offset + PDC_DATATYPE[str(msg.fields[-1].datatype)](1).nbytes
        )
        res = np.frombuffer(msg.data, np.dtype(np.int8)).reshape((-1, total_bytes))
        for field in msg.fields:
            assert field.count == 1, "If this is not case, maybe does not work"
            PDC_DATATYPE[str(field.datatype)]
            dtype = PDC_DATATYPE[str(field.datatype)]
            nbytes = dtype(1).nbytes
            data = res[:, field.offset : field.offset + nbytes]
            output[field.name] = np.frombuffer(data.copy(), np.dtype(dtype)).reshape(
                (-1)
            )

        self.store(topic, msg, (output,), msg.header)

    def store(self, topic, msg, res, header, tag="", ref_frame=None, tar_frame=None):
        p = join(self.output_folder, topic[1:].replace("/", "_"))
        os.makedirs(p, exist_ok=True)

        reference_frame = self.reference_frame
        target_frame = header.frame_id
        if ref_frame is not None:
            reference_frame = ref_frame
        if tar_frame is not None:
            target_frame = tar_frame
        try:
            trans, rot = self.tf_listener.tf_listener.lookupTransform(
                reference_frame, target_frame, header.stamp
            )
        except:
            print(f"Failed LookUp, {reference_frame}, {target_frame}")
            return False

        if res is None:
            res = (
                header.seq,
                header.stamp.nsecs,
                header.stamp.secs,
                header.frame_id,
                trans,
                rot,
            )
        else:
            res = res + (
                header.seq,
                header.stamp.nsecs,
                header.stamp.secs,
                header.frame_id,
                trans,
                rot,
            )

        with open(
            join(
                p,
                f"{secs_to_str(header.stamp.secs)}_{nsecs_to_str(header.stamp.nsecs)}{tag}.pkl",
            ),
            "wb",
        ) as handle:
            pkl.dump(res, handle, protocol=pkl.HIGHEST_PROTOCOL)

        return True

    def process(self, task_name, bag_path):
        rosbag_info_dict = get_bag_info(bag_path)
        if task_name == "trav":
            msg_handler = {
                "/crl_rzr/traversability_map_short/map_lightweight": self.store_gridmap_uint8,
                "/crl_rzr/traversability_map/map_micro": self.store_gridmap_float32,
                "/crl_rzr/traversability_map/map_short": self.store_gridmap_float32,
                "/crl_rzr/traversability_map/map_micro_ground_truth": self.store_gridmap_float32,
                "/crl_rzr/traversability_map/map_short_ground_truth": self.store_gridmap_float32,
            }
        elif task_name == "pointcloud":
            msg_handler = {"/crl_rzr/velodyne_merged_points": self.store_pointcloud}
        elif task_name == "image":
            msg_handler = {
                "/crl_rzr/multisense_front/aux/semantic_image_rect_color/compressed": self.store_compressed_image,
                "/crl_rzr/multisense_right/aux/semantic_image_rect_color/compressed": self.store_compressed_image,
                "/crl_rzr/multisense_left/aux/semantic_image_rect_color/compressed": self.store_compressed_image,
                "/crl_rzr/multisense_front/aux/semantic_image_rect_color/camera_info": self.store_camera_info,
                "/crl_rzr/multisense_left/aux/semantic_image_rect_color/camera_info": self.store_camera_info,
                "/crl_rzr/multisense_right/aux/semantic_image_rect_color/camera_info": self.store_camera_info,
            }
        else:
            raise ValueError("task_name not defined!")

        valid_topics = [k for k in msg_handler.keys()]

        try:
            with rosbag.Bag(bag_path, "r") as bag:
                pass
        except:
            subprocess.run(["rosbag", "reindex", bag_path])

        with rosbag.Bag(bag_path, "r") as bag:
            start_time = rospy.Time.from_sec(bag.get_start_time())
            end_time = rospy.Time.from_sec(bag.get_end_time())

            total_msgs = sum(
                [
                    x["messages"]
                    for x in rosbag_info_dict["topics"]
                    if x["topic"] in valid_topics
                ]
            )
            with tqdm(
                total=total_msgs,
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
            ) as pbar:
                for topic, msg, ts in bag.read_messages(
                    topics=valid_topics, start_time=start_time, end_time=end_time
                ):
                    pbar.update(1)
                    msg_handler[topic](topic, msg)

    def convert_rosbags(self, extraction_tasks):
        for task_name, bags in extraction_tasks.items():
            for j, bag_path in enumerate(bags):
                print("Task: ", task_name, f" Bag {j}/" + str(len(bags)))
                self.process(task_name, bag_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc",
        help="Store data",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="top_dir",
        default="/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    )
    args = parser.parse_args()
    # "/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    if args.target != "nan":
        result_folders = [
            os.path.join(args.target, o)
            for o in os.listdir(args.target)
            if os.path.isdir(os.path.join(args.target, o))
        ]
        result_folders = result_folders
        result_folders.sort()
    else:
        result_folders = [args.directory]
    for f in result_folders:
        print("Going to process: ", f)
    for f in result_folders:
        print("Going to process: ", f)
        RosbagConverter(f)
