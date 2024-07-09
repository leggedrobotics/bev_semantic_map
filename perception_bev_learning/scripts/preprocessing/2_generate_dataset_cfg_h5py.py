import argparse
from pathlib import Path
import os
import pickle
import numpy as np
import copy
import yaml
import torch
import pickle
import h5py
from tqdm import tqdm
from os.path import join
from perception_bev_learning.utils import get_H
from perception_bev_learning.dataset import (
    IMAGE_FRONT,
    IMAGE_LEFT,
    IMAGE_RIGHT,
    IMAGE_BACK,
)
from perception_bev_learning.dataset import (
    MAP_MICRO,
    MAP_SHORT,
    PCD_MERGED,
    ELE_MICRO,
    ELE_SHORT,
    GVOM_MICRO,
)


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


class ImageLookup:
    def __init__(self, hdf5_view: any, delta_max: float = 0.1) -> None:
        self.time = get_time(hdf5_view)
        self.delta_max = delta_max

    def get(self, float_time: float, **kwargs):
        idx = np.argmin(np.abs(self.time - float_time))

        if np.abs(self.time[idx] - float_time) < self.delta_max:
            return True, idx
        else:
            return False, -1


class GridMapLookup:
    def __init__(
        self, hdf5_view: any, delta_t: float = 10.0, delta_d: float = 5.0
    ) -> None:
        self.time = get_time(hdf5_view)
        self.position = np.array(hdf5_view["position"])
        self.delta_d = delta_d
        self.delta_t = delta_t

    def get(self, float_time: float, trans: np.ndarray, **kwargs):
        idx = np.argmin(np.linalg.norm(self.position[:, :2] - trans[:2], axis=1))
        if np.linalg.norm(self.position[idx][:2] - trans[:2]) < self.delta_d:
            if float(self.time[idx] - float_time) < self.delta_t:
                return True, idx

        return False, -1


class PointcloudLookup:
    def __init__(self, hdf5_view: any, N: int = 5) -> None:
        self.time = get_time(hdf5_view)
        self.position = np.array(hdf5_view["tf_translation"])
        self.N = N
        self.idxs = np.arange(self.time.shape[0])

    def get(self, float_time: float, **kwargs):
        idx = self.time < float_time

        res = self.idxs[idx[:, 0]].tolist()
        if len(res) >= self.N:
            return True, res[-self.N :]
        return False, -1


class ValidImages:
    def __init__(
        self, hdf5_image_view: any, minimum_delta_distance=0.5, minimum_delta_t=0.0
    ) -> None:
        N = hdf5_image_view["header_stamp_secs"].shape[0]
        self.time = get_time(hdf5_image_view)[:, 0]
        self.position = np.array(
            hdf5_image_view["tf_translation_map__sensor_origin_link"]
        )
        self.images_open_for_processing = np.ones((N,), dtype=bool)
        self.minimum_delta_distance = minimum_delta_distance
        self.minimum_delta_t = minimum_delta_t

    def get(self, n, **kwargs):
        return self.images_open_for_processing[n], n

    def mark_invalid(self, n):
        # Make the Images invalid which don't satisfy the min_delta_dist and min_delta_t thresholds
        # This is done to remove the redundant samples
        pos = self.position[n]
        dis = np.linalg.norm(self.position - pos, axis=1)
        invalid_d = dis < self.minimum_delta_distance
        self.images_open_for_processing[invalid_d] = False

        time = self.time[n]
        dis = np.abs(self.time - time)
        invalid_t = dis < self.minimum_delta_t

        self.images_open_for_processing[invalid_t] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args used in all types
    parser.add_argument("-d", "--hdf_file", type=str, help="root folder", default="nan")
    parser.add_argument("-t", "--target", type=str, default="nan")  # "nan",
    output_dir = "/data/bev_traversability/tensor01/config"

    args = vars(parser.parse_args())

    if args["target"] != "nan":
        #  if str(s).find("ignored_bags")  == -1 and str(s).find("with_data")  == -1
        h5py_files = [str(s) for s in Path(args["target"]).rglob("*with_data.h5py")]
        h5py_files.sort()
        print(h5py_files)
    else:
        h5py_files = [args["hdf_file"]]

    sequence_data = []
    for f in h5py_files:
        print(f)
        with h5py.File(f, "r") as file:
            pbar_seq = tqdm(
                total=len(file.keys()),
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
            )
            suc_count = 0
            total_count = 0

            for seq_key in file.keys():
                pbar_seq.update(1)
                seq = file[seq_key]
                print(seq.keys())
                fails = {
                    "gridmap_micro": 0,
                    "gridmap_short": 0,
                    "elevmap_micro": 0,
                    "elevmap_short": 0,
                    "image_left": 0,
                    "image_right": 0,
                    "image_back": 0,
                    "pointclouds": 0,
                    "image_front": 0,
                    "gvom_micro": 0,
                }

                if not (IMAGE_FRONT in seq.keys()):
                    print("Front camera image not available")
                    continue

                N = seq[IMAGE_FRONT]["header_stamp_secs"].shape[0]

                funcs = {
                    "image_front": ValidImages(seq[IMAGE_FRONT]),
                    "gridmap_micro": GridMapLookup(seq[MAP_MICRO], delta_d=5),
                    "gridmap_short": GridMapLookup(seq[MAP_SHORT], delta_d=5),
                    "elevmap_micro": GridMapLookup(seq[ELE_MICRO], delta_d=5),
                    "elevmap_short": GridMapLookup(seq[ELE_SHORT], delta_d=5),
                    "image_left": ImageLookup(seq[IMAGE_LEFT]),
                    "image_right": ImageLookup(seq[IMAGE_RIGHT]),
                    "image_back": ImageLookup(seq[IMAGE_BACK]),
                    "pointclouds": PointcloudLookup(seq[PCD_MERGED], N=10),
                    "gvom_micro": PointcloudLookup(seq[GVOM_MICRO], N=1),
                }

                with tqdm(
                    total=N,
                    desc="Total",
                    colour="blue",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                ) as pbar_images:
                    for j, f_image in enumerate(range(N)):
                        pbar_images.update(1)

                        sample = {}
                        float_time = get_time(seq[IMAGE_FRONT], j)
                        trans = np.array(
                            seq[IMAGE_FRONT]["tf_translation_map__sensor_origin_link"][
                                j
                            ]
                        )

                        all_true = True
                        for k, v in funcs.items():
                            suc, idx = v.get(float_time=float_time, n=j, trans=trans)
                            if suc:
                                try:
                                    sample[k] = int(idx)
                                except:
                                    sample[k] = idx
                            else:
                                fails[k] += 1
                                all_true = False
                                break

                        total_count += 1

                        # Store the sample given that everything was sucessfull
                        if all_true:
                            suc_count += 1
                            funcs["image_front"].mark_invalid(j)
                            sample["sequence_key"] = f.split("/")[-1].split(".")[0]
                            sequence_data.append(copy.deepcopy(sample))

            pbar_seq.close()

            print("Total Count: ", total_count, suc_count)
            print(fails)

    for i in range(0, len(sequence_data)):
        if i < len(sequence_data) * 0.66:
            sequence_data[i]["mode"] = "train"
        else:
            sequence_data[i]["mode"] = "val"

    with open(join(output_dir, f"dataset_config.pkl"), "wb") as handle:
        pickle.dump(sequence_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
