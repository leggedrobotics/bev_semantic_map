import argparse
from pathlib import Path
import os
import pickle
import numpy as np
import copy
import yaml
from perception_bev_learning.utils import get_H
import torch
import pickle


def pkl_load(filename: str):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def file_name_to_float_time(p: float):
    """
    This function contains some rounding erros given that float32 can not express single nanoseconds
    Args:
        p (float): filename e.g /path/1654644556_961393190.pkl

    Returns:
        (float): 1654644556.9614
    """
    return float(p.split("/")[-1][:20].replace("_", "."))


class GridMapLookup:
    def __init__(self, source_folder: str, delta_ms: float = 70000.0, delta_d: float = 5.0) -> None:
        self.potential_gridmaps = [str(s) for s in Path(source_folder).rglob(f"*.pkl")]
        timestamps = [file_name_to_float_time(s) for s in self.potential_gridmaps]
        positions = []
        for p in self.potential_gridmaps:
            gridmap = pkl_load(p)
            t = gridmap[0]["position"]
            positions.append(np.array(t))

        self.timestamps = np.array(timestamps)
        self.positions = np.stack(positions)
        self.delta_d = delta_d
        self.delta_s = delta_ms / 1000

    def get_closest_gridmap(self, float_time: float, tf_map__sensor_origin_frame: tuple):
        H_map__sensor_origin_frame = get_H(tf_map__sensor_origin_frame)
        pose = H_map__sensor_origin_frame[:2, 3].numpy()

        delta_time = np.abs(self.timestamps - float_time)
        delta_dis = np.linalg.norm(self.positions[:, :2] - pose, axis=1)

        candidate = np.argmin(delta_dis)

        if delta_dis[candidate] > self.delta_d:
            print("Distance to large ", delta_dis[candidate])
            return False, "nan", delta_dis[candidate]

        if delta_time[candidate] > self.delta_s:
            print("Time to large ", delta_time[candidate])
            return False, "nan", delta_time[candidate]

        return True, self.potential_gridmaps[candidate], delta_dis[candidate]


def get_N_last_pointclouds(source_folder: str, float_time: float, N: int = 5):
    """Returns the N pointclouds captured before a given time.

    Args:
        source_folder (str): path
        float_time (_type_): time in seconds within the rosbag
        N (int, optional): Number of pointclouds to return. Defaults to 5.

    Returns:
        suc, filenames (bool, [str]): e.g: (True, ["path/1654644556_961393190.pkl", "path/1654644556_961393191.pkl"])
    """
    # Search in window of 200ms per older pointcloud around given timestamp
    lidar_frequency = 0.1  # 100ms
    pointcloud_names = [str(s) for s in Path(source_folder).rglob(f"*")]
    pointcloud_timestamps = np.array([file_name_to_float_time(s) for s in pointcloud_names])
    m = pointcloud_timestamps < float_time
    pointcloud_timestamps = pointcloud_timestamps[m]  # only consider the older pointclouds

    if pointcloud_timestamps.shape[0] < N:
        print("To few historical pointclouds available (happens when start recording)")
        return False, "nan"

    indices = np.argsort(pointcloud_timestamps)
    pointcloud_timestamps = pointcloud_timestamps[indices]
    return_timestamps = pointcloud_timestamps[-N:]

    # validate if the oldest-considered pointcloud is in a reasonable range
    if not (return_timestamps[0] - float_time < 1.5 * lidar_frequency):
        print("Last selected pointcloud  is to old with respect to the set LiDAR record frequency.")
        return False, "nan"

    return True, np.array(pointcloud_names)[m][indices][-N:].tolist()


def path_exists_delta_t_png(filename: str, delta_ms: float = 110.0):
    """Returns the closest image with respect to a given filename

    Args:
        filename (str): e.g: "path/1654644556_961393189.png"
        delta_ms (float, optional): Maximum allowed deviation in ms from original filename. Defaults to 110.

    Returns:
        suc, filename (bool, str): e.g: (True, "path/1654644556_961393190.png")
    """

    ref_time = file_name_to_float_time(filename)
    file_names = [str(s) for s in Path(filename).parent.rglob(f"*.png")]
    file_times = [file_name_to_float_time(s) for s in file_names]
    delta_t = np.array(file_times) - np.array(ref_time)
    idx = np.argmin(np.abs(delta_t))
    min_delta_t = delta_t[idx]

    if min_delta_t < delta_ms / 1000:
        return True, file_names[idx]
    else:
        print("min_delta_t")
        return False, "nan"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args used in all types
    parser.add_argument(
        "-d",
        "--root_folder",
        type=str,
        help="root folder",
        default="/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    )
    args = vars(parser.parse_args())
    sequences = [str(s) for s in Path(args["root_folder"]).rglob("*processed")]
    dataset_root = args["root_folder"]
    config = []

    cam_topic_front = "crl_rzr_multisense_front_aux_semantic_image_rect_color_compressed"
    cam_topic_left = "crl_rzr_multisense_left_aux_semantic_image_rect_color_compressed"
    cam_topic_right = "crl_rzr_multisense_right_aux_semantic_image_rect_color_compressed"
    trav_micro = "crl_rzr_traversability_map_map_micro"
    trav_short = "crl_rzr_traversability_map_map_short"
    trav_micro_gt = "crl_rzr_traversability_map_map_micro_ground_truth"
    trav_short_gt = "crl_rzr_traversability_map_map_short_ground_truth"

    pcd_topic = "crl_rzr_velodyne_merged_points"

    for sequence in sequences:
        sequence_data = []

        gml_trav_micro = GridMapLookup(str(Path(sequence, trav_micro)), delta_d=5)
        gml_trav_micro_gt = GridMapLookup(str(Path(sequence, trav_micro_gt)), delta_d=5)
        gml_trav_short = GridMapLookup(str(Path(sequence, trav_short)), delta_d=10)
        gml_trav_short_gt = GridMapLookup(str(Path(sequence, trav_short_gt)), delta_d=10)

        image_paths = [str(s) for s in Path(sequence, cam_topic_front).rglob("*.png")]
        image_paths.sort()
        for image_path in image_paths:
            image_data = {}
            image_time = file_name_to_float_time(image_path)
            # Check if other camera topics at same timestamp exist
            suc, image_path_left = path_exists_delta_t_png(image_path.replace(cam_topic_front, cam_topic_left))
            if not suc:
                print("left camera not found")
                continue

            suc, image_path_right = path_exists_delta_t_png(image_path.replace(cam_topic_front, cam_topic_right))
            if not suc:
                print("right camera not found")
                continue

            pose_path = image_path.replace(".png", ".pkl")

            if os.path.exists(pose_path):
                tf_map__cam = pkl_load(pose_path)
            else:
                print("tf not found")
                continue

            sensor_path = image_path.replace(".png", "_map__sensor_origin_link_tf.pkl")
            if os.path.exists(sensor_path):
                tf_map__sensor_origin_link = pkl_load(sensor_path)

            suc, gridmap_path_micro, distance = gml_trav_micro.get_closest_gridmap(
                image_time, tf_map__sensor_origin_link
            )
            if not suc:
                print("gml_trav_micro")
                continue

            suc, gridmap_path_micro_gt, distance = gml_trav_micro_gt.get_closest_gridmap(
                image_time, tf_map__sensor_origin_link
            )
            if not suc:
                print("gml_trav_micro_gt")
                continue
            suc, gridmap_path_short, distance = gml_trav_short.get_closest_gridmap(
                image_time, tf_map__sensor_origin_link
            )
            if not suc:
                print("gml_trav_short")
                continue

            suc, gridmap_path_short_gt, distance = gml_trav_short_gt.get_closest_gridmap(
                image_time, tf_map__sensor_origin_link
            )
            if not suc:
                print("gml_trav_short_gt")
                continue

            suc, pointcloud_paths = get_N_last_pointclouds(str(Path(sequence, pcd_topic)), image_time)
            if not suc:
                continue
            print("working")

            image_data["image_center"] = image_path.replace(sequence + "/", "")
            image_data["image_left"] = image_path_left.replace(sequence + "/", "")
            image_data["image_right"] = image_path_right.replace(sequence + "/", "")
            image_data["gridmap_short"] = gridmap_path_short.replace(sequence + "/", "")
            image_data["gridmap_short_gt"] = gridmap_path_short_gt.replace(sequence + "/", "")
            image_data["gridmap_micro"] = gridmap_path_micro.replace(sequence + "/", "")
            image_data["gridmap_micro_gt"] = gridmap_path_micro_gt.replace(sequence + "/", "")
            image_data["pointclouds"] = [p.replace(sequence + "/", "") for p in pointcloud_paths]

            sequence_data.append(copy.deepcopy(image_data))

        config.append(
            {
                "sequence_name": sequence.replace(dataset_root + "/", ""),
                "images": len(sequence_data),
                "squence_time_in_s": file_name_to_float_time(sequence_data[-1]["image_center"])
                - file_name_to_float_time(sequence_data[0]["image_center"]),
                "data": sequence_data,
            }
        )

    with open(os.path.join(dataset_root, "dataset_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
