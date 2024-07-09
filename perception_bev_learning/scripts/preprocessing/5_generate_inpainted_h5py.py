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
import h5py
import copy
import signal
import sys

import cv2
from perception_bev_learning.utils import DatasetWriter
from perception_bev_learning.dataset import (
    IMAGE_FRONT,
    IMAGE_LEFT,
    IMAGE_RIGHT,
    IMAGE_BACK,
)
from perception_bev_learning.dataset import MAP_MICRO, MAP_SHORT, PCD_MERGED
from perception_bev_learning.preprocessing import get_h5py_files
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import NearestNDInterpolator


def get_time(hdf5_view, j=None):
    if j is None:
        return (
            np.array(hdf5_view["header_stamp_secs"])
            + np.array(hdf5_view["header_stamp_nsecs"]) * 10**-9
        )
    else:
        return (
            np.array(hdf5_view["header_stamp_secs"][j])
            + np.array(hdf5_view["header_stamp_nsecs"][j]) * 10**-9
        )


class GenerateInpainted:
    def __init__(self, h5py_file, visu, write, gridmaps=[MAP_MICRO]):
        self.visu = visu
        if self.visu:
            self.vis_pred = SimpleNumpyToRviz()
            self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")
            rospy.on_shutdown(self.shutdown_callback)

        self.write = write

        # signal.signal(signal.SIGINT, self.shutdown_callback)
        # signal.signal(signal.SIGTERM, self.shutdown_callback)

        for gridmap in gridmaps:
            self.compute(h5py_file, gridmap)

    def shutdown_callback(self, **kwargs):
        try:
            self.file.close()
        except:
            pass

        sys.exit(0)

    def compute(self, h5py_file, h5py_gridmap_key, store=False):
        self.file = h5py.File(h5py_file, "r+")

        for h5py_seq_key in self.file.keys():
            h5py_seq = self.file[h5py_seq_key]

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

            layers = [l.decode("utf-8") for l in h5py_gridmap["layers"]]
            layers += ["robust_reliable"]

            idx_rel = layers.index("reliable")
            idx_rob = layers.index("robust_reliable")
            idx_vel = layers.index("costmap_max_velocity")
            idx_wheel_risk_cvar = layers.index("wheel_risk_cvar")
            idx_wheel_risk_cvar_no_edge = layers.index("wheel_risk_cvar_no_edge")
            N, C, H, W = h5py_gridmap["data"].shape
            possible_idx = np.arange(N)
            try:
                with tqdm(
                    total=N,
                    desc="Total",
                    colour="green",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                ) as pbar:
                    # Overwrite Layers
                    gm_layers = [
                        g.decode("utf-8") for g in h5py_seq["map_micro"]["layers"]
                    ]
                    if "elevation_raw_inpainted" in gm_layers:
                        break

                    gm_layers.append("elevation_raw_inpainted")
                    gm_layers.append("wheel_risk_cvar_inpainted")
                    gm_layers.append("wheel_risk_cvar_no_edge_inpainted")

                    str_list_max_length = 50
                    compression = "lzf"

                    utf8_type = h5py.string_dtype("utf-8", str_list_max_length)
                    v = np.array(
                        [st.encode("utf-8") for st in gm_layers], dtype=utf8_type
                    )

                    data = h5py_seq["map_micro"]["data"]

                    if store:
                        del h5py_seq["map_micro"]["layers"]
                        h5py_seq["map_micro"].create_dataset(
                            "layers", data=v, compression=compression
                        )

                        # Extend the Maximum size correctly
                        maxshape = (None, None, 500, 500)
                        h5py_seq["map_micro"].create_dataset(
                            "data2",
                            data=h5py_seq["map_micro"]["data"],
                            maxshape=maxshape,
                            compression=compression,
                        )
                        del h5py_seq["map_micro"]["data"]
                        h5py_seq["map_micro"].create_dataset(
                            "data",
                            data=h5py_seq["map_micro"]["data2"],
                            maxshape=maxshape,
                            compression=compression,
                        )
                        del h5py_seq["map_micro"]["data2"]

                        print("Adjusted GridMap data to new dataset shape!")

                        data.resize((len(gm_layers)), axis=1)

                    i_er = gm_layers.index("elevation_raw")
                    i_wrc = gm_layers.index("wheel_risk_cvar")
                    i_wrcne = gm_layers.index("wheel_risk_cvar_no_edge")

                    for i in range(N):
                        for d, j in zip(
                            [data[i, i_er], data[i, i_wrc], data[i, i_wrcne]],
                            [8, 9, 10],
                        ):
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from PIL import Image
                            from perception_bev_learning.visu import get_img_from_fig

                            cmap = sns.color_palette("viridis", as_cmap=True)
                            cmap.set_bad(color="black")
                            fig = plt.figure(figsize=(5, 5))
                            plt.imshow(
                                d,
                                cmap=cmap,
                                vmin=d[~np.isnan(d)].min(),
                                vmax=d[~np.isnan(d)].max(),
                            )
                            plt.show()
                            res = get_img_from_fig(fig)
                            res.show()

                            filled_current = ~np.isnan(d)
                            indices_filled = np.where(filled_current)
                            # interp = CloughTocher2DInterpolator(indices_filled, d[filled_current])
                            interp = NearestNDInterpolator(
                                indices_filled, d[filled_current]
                            )
                            indices_interpolate = np.where(~filled_current)
                            res = interp(indices_interpolate[0], indices_interpolate[1])
                            nr = indices_interpolate[0].shape[0]
                            data_out = np.array(d)
                            data_out[
                                indices_interpolate[0], indices_interpolate[1]
                            ] = res.astype(d.dtype)

                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from PIL import Image
                            from perception_bev_learning.visu import get_img_from_fig

                            cmap = sns.color_palette("viridis", as_cmap=True)
                            cmap.set_bad(color="black")
                            fig = plt.figure(figsize=(5, 5))
                            plt.imshow(
                                data_out,
                                cmap=cmap,
                                vmin=data_out[~np.isnan(data_out)].min(),
                                vmax=data_out[~np.isnan(data_out)].max(),
                            )
                            plt.show()
                            res = get_img_from_fig(fig)
                            res.show()

                            if store:
                                data[i, j] = data_out

                        if self.visu:
                            self.vis_gt.gridmap_arr(
                                out[:, shrink:-shrink, shrink:-shrink],
                                res=res,
                                layers=layers,
                                x=110,
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
    h5py_files = get_h5py_files(default_folder="/data/bev_traversability/trial_1")

    for f in h5py_files:
        print("Following h5py file will be processed: ", f)
        GenerateInpainted(h5py_file=f, visu=False, write=True)
