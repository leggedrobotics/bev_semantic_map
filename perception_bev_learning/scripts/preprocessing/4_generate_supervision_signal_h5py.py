import sys
import os
from pathlib import Path
import argparse
import rospy
import numpy as np
from tqdm import tqdm

from os.path import join
import cv2
from perception_bev_learning.ros import SimpleNumpyToRviz
from perception_bev_learning.utils import load_pkl
import warnings
from os.path import join
import time
import torch
from torchvision.transforms.functional import affine
from tqdm import tqdm
import h5py
import copy
import signal
import sys
from torchvision.transforms.functional import pad

from perception_bev_learning.utils import DatasetWriter

from perception_bev_learning.dataset.h5py_keys import TRAV_LAYERS
from perception_bev_learning.preprocessing import get_h5py_files

HW_ext = (720, 720)  # Hardcoded value for grids of (500,500)


def get_time(hdf5_view, j=None):
    if j is None:
        return (
            np.array(hdf5_view["header_stamp_secs"])
            + np.array(hdf5_view["header_stamp_nsecs"]) * 10 ** -9
        )
    else:
        return (
            np.array(hdf5_view["header_stamp_secs"][j])
            + np.array(hdf5_view["header_stamp_nsecs"][j]) * 10 ** -9
        )


def mean(measurment, map1, map2, *args, **kwargs):
    m = ~np.isnan(measurment)
    map1[m] += measurment[m]
    map2[m] += 1
    return map1, map2


def latest(measurment, map1, map2, *args, **kwargs):
    m = ~np.isnan(measurment)
    map1[m] = measurment[m]
    map2[m] = 1
    return map1, map2


def latest_reliable(measurment, map1, map2, reliable, *args, **kwargs):

    meas_valid = ~np.isnan(measurment) * (reliable > 0.5)
    # measurment valid
    map1[meas_valid] = measurment[meas_valid]
    map2[meas_valid] = 1

    return map1, map2


def maximum(measurment, map1, map2, *args, **kwargs):
    meas_valid = ~np.isnan(measurment)

    map_invalid = np.isnan(map1)

    # map cell invalid but measurment valid
    map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

    # both inputs valid and measurment is greater
    m_both_valid = meas_valid * ~map_invalid
    m_measurment_greater = np.nan_to_num(measurment) > np.nan_to_num(map1)
    map1[m_both_valid * m_measurment_greater] = measurment[
        m_both_valid * m_measurment_greater
    ]
    map2[:, :] = 1
    return map1, map2


def maximum_reliable(measurment, map1, map2, reliable, *args, **kwargs):
    meas_valid = ~np.isnan(measurment) * (reliable > 0.5)

    map_invalid = np.isnan(map1)

    # map cell invalid but measurment valid
    map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

    # both inputs valid and measurment is greater
    m_both_valid = meas_valid * ~map_invalid
    m_measurment_greater = np.nan_to_num(measurment) > np.nan_to_num(map1)
    map1[m_both_valid * m_measurment_greater] = measurment[
        m_both_valid * m_measurment_greater
    ]
    map2[:, :] = 1
    return map1, map2


def minimum(measurment, map1, map2, *args, **kwargs):
    meas_valid = ~np.isnan(measurment)

    map_invalid = np.isnan(map1)

    # map cell invalid but measurment valid
    map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

    # both inputs valid and measurment is smaller
    m_both_valid = meas_valid * ~map_invalid
    m_measurment_greater = np.nan_to_num(measurment) < np.nan_to_num(map1)
    map1[m_both_valid * m_measurment_greater] = measurment[
        m_both_valid * m_measurment_greater
    ]
    map2[:, :] = 1
    return map1, map2


def minimum_reliable(measurment, map1, map2, reliable, *args, **kwargs):
    meas_valid = ~np.isnan(measurment) * (reliable > 0.5)

    map_invalid = np.isnan(map1)

    # map cell invalid but measurment valid
    map1[map_invalid * meas_valid] = measurment[map_invalid * meas_valid]

    # both inputs valid and measurment is greater
    m_both_valid = meas_valid * ~map_invalid
    m_measurment_greater = np.nan_to_num(measurment) < np.nan_to_num(map1)
    map1[m_both_valid * m_measurment_greater] = measurment[
        m_both_valid * m_measurment_greater
    ]
    map2[:, :] = 1
    return map1, map2


class GenerateGTSupervision:
    def __init__(self, h5py_file, visu, write, gridmaps=[MAP_MICRO]):
        self.visu = visu
        if self.visu:
            self.vis_pred = SimpleNumpyToRviz()
            self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")
            rospy.on_shutdown(self.shutdown_callback)

        self.write = write

        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

        for gridmap in gridmaps:
            self.compute(h5py_file, gridmap)

    def shutdown_callback(self):
        try:
            self.file.close()
        except:
            pass

        sys.exit(0)

    def compute(
        self, h5py_file, h5py_gridmap_key, max_accumulation_time=10, override=True
    ):
        self.file = h5py.File(h5py_file, "r+")

        if self.write:
            dataset_writer = DatasetWriter(h5py_file, open_file=self.file)

        for h5py_seq_key in self.file.keys():
            h5py_seq = self.file[h5py_seq_key]

            if override:
                try:
                    del h5py_seq[h5py_gridmap_key + "_gt"]
                except:
                    print("GT does not exist")
                    pass
            else:
                if h5py_gridmap_key + "_gt" in h5py_seq.keys():
                    N1 = h5py_seq[h5py_gridmap_key]["header_seq"].shape[0]
                    N2 = h5py_seq[h5py_gridmap_key + "_gt"]["header_seq"].shape[0]
                    if N1 == N2:
                        print(
                            "Did not further process sequence given that GT already exists!"
                        )
                        continue

            h5py_gridmap = h5py_seq[h5py_gridmap_key]
            t = get_time(h5py_gridmap)

            # Ensure data is sorted according to timestamp
            t_ls = t[:, 0].tolist()
            t_sort = copy.deepcopy(t_ls)
            t_sort.sort()
            assert (
                t_ls == t_sort
            ), "GridMap data is not sorted correctly according to timestamp"

            pose = np.array(h5py_gridmap["position"])
            timestamps = np.array(t[:, 0])
            H_c, W_c = (
                int(h5py_gridmap["data"].shape[2] / 2),
                int(h5py_gridmap["data"].shape[3] / 2),
            )
            length = np.array(h5py_gridmap["length"])
            res = np.array(h5py_gridmap["resolution"])
            shape = (int(length[0] / res), int(length[1] / res))

            # TODO (Manthan) Get rid of the fusions dictionary here and instead put it in h5py_keys file
            # For now, we need to modify all the layers correctly as per the config
            # For now, we will just make sure that all the layer fusion functions exist when compared to the TRAV layers
            fusions = {
                "elevation_raw": latest,
                "num_points": maximum,
                "reliable": maximum,
                "reliable_filled": latest,
                "confidence": latest,
                "elevation": latest,
                "unknown": latest,
                "wheel_risk": latest,
                "max_velocity": latest,
                "robust_reliable": maximum,
            }
            for k in TRAV_LAYERS:
                if not k in fusions.keys():
                    print(f"{k} is missing")

            layers = [l.decode("utf-8") for l in h5py_gridmap["layers"]]
            layers += ["robust_reliable"]

            idx_rel = layers.index("reliable")
            idx_rob = layers.index("robust_reliable")
            idx_vel = layers.index("max_velocity")
            idx_wheel_risk = layers.index("wheel_risk")
            idx_ele = layers.index("elevation")
            # idx_wheel_risk_cvar_no_edge = layers.index("wheel_risk_cvar_no_edge")
            N, C, H, W = h5py_gridmap["data"].shape
            H_ext, W_ext = HW_ext
            pad_val = [int((H_ext - H) / 2), int((W_ext - W) / 2)]
            possible_idx = np.arange(N)
            try:
                with tqdm(
                    total=N,
                    desc="Total",
                    colour="green",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                ) as pbar:
                    for i in range(N):
                        if not override:
                            if h5py_gridmap_key + "_gt" in h5py_seq.keys():
                                if (
                                    h5py_seq[h5py_gridmap_key + "_gt"][
                                        "header_seq"
                                    ].shape[0]
                                    > i
                                ):
                                    if bool(
                                        h5py_seq[h5py_gridmap_key + "_gt"][
                                            "header_seq"
                                        ][i]
                                        == h5py_seq[h5py_gridmap_key]["header_seq"][i]
                                    ):
                                        pbar.update(1)
                                        print("already processed continue")
                                        continue

                        m_pose = (
                            np.linalg.norm(pose - pose[i], ord=1, axis=1) < length[0]
                        )
                        m_time = (timestamps - timestamps[i]) < max_accumulation_time
                        m = m_pose * m_time

                        # map1 = np.zeros((C + 1, H, W), dtype=np.float32)
                        # map1[-1, :, :] = 10000
                        # map2 = np.zeros((C + 1, H, W), dtype=np.float32)

                        # Increase the GT Map Size to root 2 times
                        # map1 = np.full((C + 1, H_ext, W_ext), np.nan, dtype=np.float32)
                        map1 = np.zeros((C + 1, H_ext, W_ext), dtype=np.float32)
                        # map1[-1, :, :] = 10000
                        map2 = np.zeros((C + 1, H_ext, W_ext), dtype=np.float32)
                        H_c = int(H_ext / 2)
                        W_c = int(W_ext / 2)

                        reference_pose = pose[i]
                        subdiv = max(1, int(possible_idx[m].shape[0] / 40))

                        for k in possible_idx[m][::subdiv]:
                            data = np.array(h5py_gridmap["data"][k])
                            data = torch.from_numpy(data)[None]

                            # Pad the data according to the ext
                            data = pad(data, pad_val, torch.nan, "constant")

                            shift = -(reference_pose - pose[k]) / res
                            sh = [shift[1], shift[0]]
                            try:
                                data_shifted = affine(
                                    data,
                                    angle=0,
                                    translate=sh,
                                    scale=1,
                                    shear=0,
                                    center=(H_c, W_c),
                                    fill=torch.nan,
                                )[0].numpy()
                            except:
                                data_shifted = affine(
                                    data,
                                    angle=0,
                                    translate=sh,
                                    scale=1,
                                    shear=0,
                                    center=(H_c, W_c),
                                    fill=torch.nan,
                                )[0].numpy()

                            # Create robust reliable layer
                            reliable = data_shifted[idx_rel]
                            reliable[np.isnan(reliable)] = 0

                            size = (5, 5)
                            shape = cv2.MORPH_RECT
                            kernel = cv2.getStructuringElement(shape, size)
                            robust_reliable = cv2.erode(reliable, kernel)
                            data_shifted = np.concatenate(
                                [data_shifted, robust_reliable[None]], axis=0
                            )

                            # out = np.zeros_like( data_shifted[idx_wheel_risk_cvar] )
                            # out[:,:] = np.nan
                            # mask = np.isnan(data_shifted[idx_wheel_risk_cvar_no_edge])

                            # out, _ = minimum(data_shifted[idx_wheel_risk_cvar_no_edge], out, np.zeros((1,1)))
                            # out, _ = minimum(np.nan_to_num(data_shifted[idx_wheel_risk_cvar]), out, np.zeros((1,1)))
                            # out[mask] = 0.1

                            # # There is a weird error of some untraversable squares at the outer corner
                            # data_shifted[idx_wheel_risk_cvar_no_edge] = out

                            m = robust_reliable < 0.7

                            data_shifted[idx_wheel_risk][m] = np.nan
                            # data_shifted[idx_ele][m] = np.nan

                            # minimum(data_shifted[idx_wheel_risk_cvar_no_edge].copy(), data_shifted[idx_wheel_risk_cvar].copy(), np.zeros((1,1)))

                            for j, l in enumerate(layers):
                                map1[j], map2[j] = fusions[l](
                                    data_shifted[j], map1[j], map2[j], robust_reliable
                                )

                            if self.visu:
                                shrink = 1
                                self.vis_pred.gridmap_arr(
                                    data_shifted[:, shrink:-shrink, shrink:-shrink],
                                    res=res,
                                    layers=layers,
                                )

                        ma = map1[idx_vel] == 10000
                        if ma.sum() != 0:
                            map1[idx_vel][ma] = 0
                            map2[idx_vel][ma] = 1

                        out = map1.copy()
                        for j, l in enumerate(layers):
                            prediction_invalid = map2 == 0
                            map1[prediction_invalid] = np.nan

                            m = ~np.isnan(map1)

                            if m.sum() == 0:
                                out[m] = np.nan
                            else:
                                out[m] = map1[m] / map2[m]

                        static = {
                            "layers": layers,
                            "resolution": np.array(h5py_gridmap["resolution"]),
                            "length": np.array(
                                h5py_gridmap["length"]
                            ),  # TODO This would be changed to the new value
                            "header_frame_id": "map",
                        }
                        dynamic = {
                            "data": out,
                            "header_seq": h5py_gridmap["header_seq"][i],
                            "header_stamp_nsecs": h5py_gridmap["header_stamp_nsecs"][i],
                            "header_stamp_secs": h5py_gridmap["header_stamp_secs"][i],
                            "orientation_xyzw": h5py_gridmap["orientation_xyzw"][i],
                            "position": h5py_gridmap["position"][i],
                            "tf_rotation_xyzw": h5py_gridmap["tf_rotation_xyzw"][i],
                            "tf_translation": h5py_gridmap["tf_translation"][i],
                        }
                        if self.write:
                            dataset_writer.add_static(
                                sequence=h5py_seq_key,
                                fieldname=h5py_gridmap_key + "_gt",
                                data_dict=static,
                            )
                            dataset_writer.add_data(
                                sequence=h5py_seq_key,
                                fieldname=h5py_gridmap_key + "_gt",
                                data_dict=dynamic,
                            )

                        if self.visu:
                            self.vis_gt.gridmap_arr(
                                out[:, shrink:-shrink, shrink:-shrink],
                                res=res,
                                layers=layers,
                                x=0,
                            )

                        pbar.update(1)
            except Exception as e:
                self.file.close()
                print(e)
                raise Exception(e)
            self.file.close()

    def __del__(self):
        self.file.close()


if __name__ == "__main__":
    h5py_files = get_h5py_files(default_folder="nan")
    print(h5py_files)
    for f in h5py_files:
        print("Following h5py file will be processed: ", f)
        GenerateGTSupervision(h5py_file=f, visu=True, write=True)
