import os
from pathlib import Path
import argparse
import rospy
import numpy as np
import sys
from tqdm import tqdm
import rosbag
from os.path import join
from cv_bridge import CvBridge
import subprocess
import warnings
from perception_bev_learning.preprocessing import BagTfTransformerWrapper, get_bag_info
from perception_bev_learning.preprocessing import fix_rosbags, merge_bags_single
from perception_bev_learning.preprocessing import (
    PDC_DATATYPE,
    suppress_TF_REPEATED_DATA,
    msg_to_se3,
)
from perception_bev_learning.utils import DatasetWriter, convert_gridmap_float32
from pytictac import CpuTimer
from perception_bev_learning.dataset.h5py_keys import *

suppress_TF_REPEATED_DATA()


class RosbagConverter:
    def __init__(self, directory, dataset_writer, only_header, fix_rosbags_=False):
        self.only_header = only_header
        if fix_rosbags_:
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
        gvom_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find("_bev_gvom_") != -1
        ]

        output_bag_tf = join(directory, "merged_tf.bag")

        if len(tf_rosbags) > 1:
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
        else:
            output_bag_tf = tf_rosbags[0]

        print("Start Loading Rosbag Odometry")
        self.tf_listener = BagTfTransformerWrapper(output_bag_tf)

        self.reference_frame = "crl_rzr/map"

        if not self.only_header:
            self.bridge = CvBridge()

        extraction_tasks = {
            "pointcloud": pointcloud_rosbags,
            "gvom": gvom_rosbags,
            "image": image_rosbags,
            "trav": trav_rosbags,
        }

        self.dataset_writer = dataset_writer
        self.camera_infos_stored = {}

        for task_name, bags in extraction_tasks.items():
            bags.sort()
            for j, bag_path in enumerate(bags):
                print("Task: ", task_name, f" Bag {j}/" + str(len(bags) - 1))
                self.process(task_name, bag_path)

    def store_camera_info(self, topic, msg, current, total):
        fieldname = topic[1:].replace("/", "_")
        try:
            if self.camera_infos_stored[fieldname]:
                return
        except:
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
                    camera_info[k] = np.array(list(v))

            self.dataset_writer.add_static(self.sequence, fieldname, camera_info)
            self.camera_infos_stored[fieldname] = True

    def store_compressed_image(self, topic, msg, current, total):
        if self.only_header:
            res_dict = {}
        else:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            res_dict = {"image": cv_img}

        tf_dict_map__camera, suc1 = self.get_tf_and_header_dict(msg.header)
        tf_dict_map__base_link, suc2 = self.get_tf_and_header_dict(
            msg.header, tar_frame="crl_rzr/base_link"
        )
        tf_dict_map__sensor_origin_link, suc3 = self.get_tf_and_header_dict(
            msg.header, tar_frame="crl_rzr/sensor_origin_link"
        )
        tf_dict_map__odom, suc4 = self.get_tf_and_header_dict(
            msg.header, tar_frame="crl_rzr/odom"
        )

        if suc1 and suc2 and suc3 and suc4:
            fieldname = topic[1:].replace("/", "_")
            res_dict.update(tf_dict_map__camera)

            res_dict["tf_translation_map__odom"] = tf_dict_map__odom["tf_translation"]
            res_dict["tf_rotation_xyzw_map__odom"] = tf_dict_map__odom[
                "tf_rotation_xyzw"
            ]

            res_dict["tf_translation_map__base_link"] = tf_dict_map__base_link[
                "tf_translation"
            ]
            res_dict["tf_rotation_xyzw_map__base_link"] = tf_dict_map__base_link[
                "tf_rotation_xyzw"
            ]

            res_dict[
                "tf_translation_map__sensor_origin_link"
            ] = tf_dict_map__sensor_origin_link["tf_translation"]
            res_dict[
                "tf_rotation_xyzw_map__sensor_origin_link"
            ] = tf_dict_map__sensor_origin_link["tf_rotation_xyzw"]

            static_keys = ["header_frame_id"]
            static_dict = {k: v for k, v in res_dict.items() if k in static_keys}
            self.dataset_writer.add_static(self.sequence, fieldname, static_dict)
            dynamic_dict = {k: v for k, v in res_dict.items() if k not in static_keys}
            self.dataset_writer.add_data(
                self.sequence, fieldname, dynamic_dict, current, total
            )

    def store_gridmap_float32(self, topic, msg, current, total):
        res_dict = convert_gridmap_float32(msg, self.only_header)
        tf_dict, suc = self.get_tf_and_header_dict(msg.info.header)
        fieldname = topic[1:].replace("/", "_")
        if not suc:
            print("TF lookup failed ", topic)
            return

        res_dict.update(tf_dict)
        static_keys = ["layers", "resolution", "length", "header_frame_id"]
        static_dict = {k: v for k, v in res_dict.items() if k in static_keys}
        self.dataset_writer.add_static(self.sequence, fieldname, static_dict)
        dynamic_dict = {
            k: v
            for k, v in res_dict.items()
            if k not in static_keys and k != "basic_layers"
        }
        with CpuTimer("store", verbose=False):
            self.dataset_writer.add_data(
                self.sequence, fieldname, dynamic_dict, current, total
            )

    def store_pointcloud(self, topic, msg, current, total):
        res_dict = {}
        fieldname = topic[1:].replace("/", "_")
        if not self.only_header:
            total_bytes = (
                msg.fields[-1].offset
                + PDC_DATATYPE[str(msg.fields[-1].datatype)](1).nbytes
            )
            res = np.frombuffer(msg.data, np.dtype(np.int8)).reshape((-1, total_bytes))
            for field in msg.fields:
                assert field.count == 1, "If this is not case, maybe does not work"
                PDC_DATATYPE[str(field.datatype)]
                dtype = PDC_DATATYPE[str(field.datatype)]
                nbytes = dtype(1).nbytes
                data = res[:, field.offset : field.offset + nbytes]
                res_dict[field.name] = np.frombuffer(
                    data.copy(), np.dtype(dtype)
                ).reshape((-1))

            res_dict["valid"] = np.ones((res_dict[field.name].shape[0],), dtype=bool)

        tf_dict, suc = self.get_tf_and_header_dict(msg.header)

        if not suc:
            print("TF lookup failed ", topic)
            return

        # res_dict.update(tf_dict)
        static_keys = ["header_frame_id"]
        static_dict = {k: v for k, v in tf_dict.items() if k in static_keys}
        self.dataset_writer.add_static(self.sequence, fieldname, static_dict)

        dynamic_dict = {k: v for k, v in tf_dict.items() if k not in static_keys}
        self.dataset_writer.add_data(
            self.sequence, fieldname, dynamic_dict, current, total
        )
        if not self.only_header:
            self.dataset_writer.add_pointcloud(
                self.sequence, fieldname, res_dict, current, total
            )

    def get_tf_and_header_dict(self, header, ref_frame=None, tar_frame=None):
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
            print(f"Failed LookUp, {reference_frame}, {target_frame} ")
            return {}, False

        res = {}
        res["header_seq"] = header.seq
        res["header_stamp_nsecs"] = header.stamp.nsecs
        res["header_stamp_secs"] = header.stamp.secs
        res["header_frame_id"] = header.frame_id
        res["tf_translation"] = np.array(trans)
        res["tf_rotation_xyzw"] = np.array(rot)

        return res, True

    def process(self, task_name, bag_path):
        rosbag_info_dict = get_bag_info(bag_path)
        if task_name == "trav":
            msg_handler = {
                MAP_MICRO_TOPIC: self.store_gridmap_float32,
                MAP_SHORT_TOPIC: self.store_gridmap_float32,
                ELE_MICRO_TOPIC: self.store_gridmap_float32,
                ELE_SHORT_TOPIC: self.store_gridmap_float32,
            }
        elif task_name == "pointcloud":
            msg_handler = {PCD_MERGED_TOPIC: self.store_pointcloud}
        elif task_name == "gvom":
            msg_handler = {GVOM_MICRO_TOPIC: self.store_pointcloud}
        elif task_name == "image":
            msg_handler = {
                IMAGE_FRONT_TOPIC: self.store_compressed_image,
                IMAGE_BACK_TOPIC: self.store_compressed_image,
                IMAGE_LEFT_TOPIC: self.store_compressed_image,
                IMAGE_RIGHT_TOPIC: self.store_compressed_image,
                CAMERA_INFO_BACK_TOPIC: self.store_camera_info,
                CAMERA_INFO_FRONT_TOPIC: self.store_camera_info,
                CAMERA_INFO_LEFT_TOPIC: self.store_camera_info,
                CAMERA_INFO_RIGHT_TOPIC: self.store_camera_info,
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
            self.sequence = bag_path.split("/")[-2]
            try:
                start_time = rospy.Time.from_sec(bag.get_start_time())
            except:
                print("Skiped the bag given that it is empty")
                return

            end_time = rospy.Time.from_sec(bag.get_end_time())

            total_msgs = sum(
                [
                    x["messages"]
                    for x in rosbag_info_dict["topics"]
                    if x["topic"] in valid_topics
                ]
            )
            total_counts = {
                i["topic"] + "_total_count": i["messages"]
                for i in rosbag_info_dict["topics"]
            }
            current_counts = {
                i["topic"] + "_current_count": 0 for i in rosbag_info_dict["topics"]
            }

            with tqdm(
                total=total_msgs,
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                file=sys.stdout,
                leave=True,
            ) as pbar:
                for topic, msg, ts in bag.read_messages(
                    topics=valid_topics, start_time=start_time, end_time=end_time
                ):
                    with CpuTimer("Process message time", verbose=False):
                        pbar.update(1)
                        msg_handler[topic](
                            topic,
                            msg,
                            current_counts[topic + "_current_count"],
                            total_counts[topic + "_total_count"],
                        )
                        current_counts[topic + "_current_count"] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y1_d2_t1_A5_to_A4_Tue_Jun__7_19-21-15_2022_utc",
        help="Store data",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="top_dir",
        default="nan",  # "/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    )
    args = parser.parse_args()

    if args.target != "nan":
        result_folders = [
            join(args.target, o)
            for o in os.listdir(args.target)
            if os.path.isdir(join(args.target, o))
        ]
        result_folders = result_folders
        result_folders.sort()
    else:
        result_folders = [args.directory]

    for f in result_folders:
        print("Following folder are selected for processing: ", f)

    for f in result_folders:
        print("Going to process: ", f)
        name = f.split("/")[-1]
        dataset_file = join(str(Path(result_folders[0]).parent), f"{name}.h5py")
        if not os.path.exists(dataset_file):
            dataset_writer = DatasetWriter(dataset_file)
            RosbagConverter(f, dataset_writer, only_header=True)
            dataset_writer.close()
        else:
            print("H5 file already exists")

    print("Finished processing all folders")
