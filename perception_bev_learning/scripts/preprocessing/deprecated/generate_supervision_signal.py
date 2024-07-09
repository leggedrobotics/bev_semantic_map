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
from perception_bev_learning.ros import SimpleNumpyToRviz
from perception_bev_learning.utils import load_pkl
import warnings
from os.path import join
import time
import torch
from torchvision.transforms.functional import affine
from tqdm import tqdm

# TODO 1 get all gridmap infos, timestamp + pose
# TODO iterate over all gridmaps and get those gridmaps that cover the same FoV
# TODO publish the current gridmap in RvIZ
# TODO publish the accumulated gridmap in RvIZ


class GenerateGTSupervision:
    def __init__(self, directory, gridmaps=["crl_rzr_traversability_map_map_micro"]):
        self.vis_pred = SimpleNumpyToRviz()
        self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")

        for gridmap in gridmaps:
            self.compute(directory, gridmap)

    def compute(self, directory, gridmap, max_accumulation_time=50):
        gridmap_paths = [
            str(s) for s in Path(directory, "processed", gridmap).rglob("*.pkl") if str(s)[-6:] != "gt.pkl"
        ]
        gridmap_paths.sort()
        pose = []
        timestamps = []

        # gridmap_paths = gridmap_paths[::]

        for p in gridmap_paths:
            grid_map = load_pkl(p)
            pose.append(grid_map[0]["position"])
            timestamps.append(float(grid_map[3]) + float(grid_map[2]) * (10**-10))

        pose = np.stack(pose)
        timestamps = np.array(timestamps)
        H_c, W_c = int(grid_map[0]["data"].shape[1] / 2), int(grid_map[0]["data"].shape[2] / 2)
        length = grid_map[0]["length"]
        res = grid_map[0]["resolution"]
        shape = (int(length[0] / res), int(length[1] / res))

        def mean(measurment, map1, map2):
            m = ~np.isnan(measurment)
            map1[m] += measurment[m]
            map2[m] += 1

            return map1, map2

        def latest(measurment, map1, map2):
            m = ~np.isnan(measurment)
            map1[m] += measurment[m]
            map2[:, :] = 1
            return map1, map2

        def maximum(measurment, map1, map2):
            meas_valid = ~np.isnan(measurment)

            map_invalid = np.isnan(map1)

            # map cell invalid but measurment valid
            map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

            # both inputs valid and measurment is greater
            m_both_valid = meas_valid * ~map_invalid
            m_measurment_greater = np.nan_to_num(measurment) > np.nan_to_num(map1)
            map1[m_both_valid * m_measurment_greater] = measurment[m_both_valid * m_measurment_greater]
            map2[:, :] = 1
            return map1, map2

        def minimum(measurment, map1, map2):
            meas_valid = ~np.isnan(measurment)

            map_invalid = np.isnan(map1)

            # map cell invalid but measurment valid
            map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

            # both inputs valid and measurment is greater
            m_both_valid = meas_valid * ~map_invalid
            m_measurment_greater = np.nan_to_num(measurment) < np.nan_to_num(map1)
            map1[m_both_valid * m_measurment_greater] = measurment[m_both_valid * m_measurment_greater]
            map2[:, :] = 1
            return map1, map2

        fusions = {
            "elevation_raw": mean,
            "num_points": maximum,
            "reliable": maximum,
            "elevation": mean,
            "unknown": maximum,
            "wheel_risk_cvar": maximum,
            "costmap_max_velocity": minimum,
        }

        with tqdm(
            total=len(gridmap_paths),
            desc="Total",
            colour="green",
            position=1,
            bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
        ) as pbar:
            for j, path in enumerate(gridmap_paths):
                m_pose = np.linalg.norm(pose - pose[j], ord=1, axis=1) < length[0]
                m_time = (timestamps - timestamps[j]) < max_accumulation_time
                m = m_pose * m_time

                map1 = np.zeros_like(grid_map[0]["data"], dtype=np.float32)
                map1[-1, :, :] = 10000
                map2 = np.zeros_like(grid_map[0]["data"], dtype=np.float32)

                reference_pose = pose[j]

                for k, p_in in enumerate(np.array(gridmap_paths)[m]):
                    grid_map = load_pkl(p_in)
                    data = grid_map[0]["data"]

                    data = torch.from_numpy(data)[None]

                    shift = -(reference_pose - grid_map[0]["position"]) / res
                    sh = [shift[1], shift[0]]
                    data_shifted = affine(
                        data, angle=0, translate=sh, scale=1, shear=0, center=(H_c, W_c), fill=torch.nan
                    )[0].numpy()

                    for j, l in enumerate(grid_map[0]["layers"]):
                        map1[j], map2[j] = fusions[l](data_shifted[j], map1[j], map2[j])

                    shrink = 1
                    # self.vis_pred.gridmap_arr(data_shifted[:, shrink:-shrink, shrink:-shrink], res =res, layers=grid_map[0]["layers"])

                out = map1.copy()
                for j, l in enumerate(grid_map[0]["layers"]):
                    m = ~np.isnan(map1)
                    out[m] = map1[m] / map2[m]

                # self.vis_gt.gridmap_arr(out[:, shrink:-shrink, shrink:-shrink], res =res, layers=grid_map[0]["layers"], x= 110)

                gridmap_origin = load_pkl(path)
                gridmap_origin[0]["data"] = out

                with open(path.replace(".pkl", "_gt.pkl"), "wb") as handle:
                    pkl.dump(gridmap_origin, handle, protocol=pkl.HIGHEST_PROTOCOL)

                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/media/jonfrey/Elements/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc",
        help="Store data",
    )
    args = parser.parse_args()
    GenerateGTSupervision(args.directory)
