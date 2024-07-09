from perception_bev_learning.utils import load_yaml, load_env
import pickle
from os.path import join
from perception_bev_learning.utils import DatasetWriter
import numpy as np
import h5py
from perception_bev_learning.dataset import IMAGE_FRONT, IMAGE_LEFT, IMAGE_RIGHT, IMAGE_BACK
from perception_bev_learning.dataset import MAP_MICRO, MAP_SHORT, PCD_MERGED
import os
import cv2
import copy
from pytictac import CpuTimer
import argparse

W, H = 640, 396
parser = argparse.ArgumentParser()
# args used in all types
parser.add_argument(
    "-s",
    "--subsample",
    type=int,
    help="Subsample",
    default=1,
)
args = vars(parser.parse_args())

subsample = args["subsample"]

pcd_history = 1
compression = "lzf"
print("subsample", subsample)
print("pcd_history", pcd_history)

# Iterate over all .h5py files defined in the orginial dataset_config.pkl
# Create for each .h5py a new one appended with _subsample_{subsample} and copy all data which is needed.
# Create a new dataset_config.pkl with the new subsampled data.

# Features: Currently the images are rescaled to W,H
# The camera_info_new is adjusted accordingly
# The data can be subsampled by a factor of subsample
# For the pointclouds it can be defined how many history frames are used - in the raw data we set this number to 10


def add_running_datasets(running_datasets, s, h5py_k, h5py_new_handles, h5py_handles, used_datapoints):
    for res in running_datasets:
        running_key, select = res
        v = h5py_handles[s][s][h5py_k][running_key]
        if select:
            v = v[used_datapoints]
            h5py_new_handles[s][s][h5py_k].create_dataset(running_key, data=v, compression=compression)
        else:
            h5py_new_handles[s].copy(h5py_handles[s][s][h5py_k][running_key], s + "/" + h5py_k + "/" + running_key)


root_dir = load_env()["dataset_root_dir"]
with open(join(root_dir, "dataset_config.pkl"), "rb") as handle:
    dataset_config = pickle.load(handle)

dataset_config = dataset_config[::subsample]
dataset_config_new = copy.deepcopy(dataset_config)
unique_sequence_keys = np.unique([np.array(d["sequence_key"]) for d in dataset_config]).tolist()
for s in unique_sequence_keys:
    path = join(root_dir, s + f"_subsample_{subsample}.h5py")
    os.system(f"rm {path}")

print(unique_sequence_keys)
h5py_handles = {}

unique_sequence_keys_working = []
for s in unique_sequence_keys:
    try:
        h5py_handles[s] = h5py.File(join(root_dir, s + ".h5py"), "r")
        unique_sequence_keys_working.append(s)
    except:
        print(f"Try to open: failed: {s}")

# h5py_handles = {s: h5py.File(join(root_dir, s + ".h5py"), "r") for s in unique_sequence_keys}

h5py_new_handles = {
    s: h5py.File(join(root_dir, s + f"_subsample_{subsample}.h5py"), "w") for s in unique_sequence_keys_working
}
h5py_keys = [IMAGE_FRONT, IMAGE_BACK, IMAGE_LEFT, IMAGE_RIGHT, PCD_MERGED, MAP_MICRO, "map_micro_gt"]
keys = ["image_front", "image_back", "image_left", "image_right", "pointclouds", "gridmap_micro", "gridmap_gt_micro"]

for s in unique_sequence_keys_working:
    # remove all data which is not needed

    with CpuTimer("Rescale camera info"):
        for k, h5py_k in zip(keys[:4], [IMAGE_FRONT, IMAGE_BACK, IMAGE_LEFT, IMAGE_RIGHT]):
            factor = h5py_handles[s][s][h5py_k]["image"].shape[2] / W
            h5py_k = h5py_k.replace("compressed", "camera_info")

            camera_info = h5py_handles[s][s][h5py_k]

            if s not in h5py_new_handles[s]:
                h5py_new_handles[s].create_group(s)

            # origin to destination copy
            h5py_new_handles[s].copy(h5py_handles[s][s][h5py_k], s + "/" + h5py_k)
            camera_info_new = h5py_new_handles[s][s][h5py_k]
            camera_info = h5py_handles[s][s][h5py_k]

            camera_info_new["K"][:6] = np.array(camera_info["K"][:6]) / factor
            camera_info_new["P"][:8] = np.array(camera_info["P"][:8]) / factor
            camera_info_new["P"][-1] = np.array(camera_info["P"][-1]) / factor
            camera_info_new["height"][:] = int(camera_info["height"][0] / factor)
            camera_info_new["width"][:] = int(camera_info["width"][0] / factor)

    for k, h5py_k in zip(keys, h5py_keys):
        used_datapoints = np.stack(
            [np.array(d[k.replace("_gt", "")]) for d in dataset_config if d["sequence_key"] == s]
        )
        mapping = np.full(used_datapoints.max() + 1, -1, dtype=np.int32)
        used_datapoints = np.unique(used_datapoints)
        mapping[used_datapoints] = np.arange(used_datapoints.shape[0])

        # Copy statis data
        if not h5py_k in h5py_new_handles[s][s].keys():
            h5py_new_handles[s][s].create_group(h5py_k)

        with CpuTimer(f"{k}"):
            if k.find("image") != -1:
                # overwrite the dataset config
                for j in range(len(dataset_config_new)):
                    if dataset_config_new[j]["sequence_key"] == s:
                        dataset_config_new[j][k] = mapping[dataset_config_new[j][k]]

                # Write messages other data
                running_datasets = [
                    ("header_frame_id", False),
                    ("tf_rotation_xyzw", True),
                    ("tf_rotation_xyzw_map__base_link", True),
                    ("tf_rotation_xyzw_map__odom", True),
                    ("tf_rotation_xyzw_map__sensor_origin_link", True),
                    ("tf_translation", True),
                    ("tf_translation_map__base_link", True),
                    ("tf_translation_map__odom", True),
                    ("tf_translation_map__sensor_origin_link", True),
                    ("header_stamp_secs", True),
                    ("header_stamp_nsecs", True),
                    ("header_seq", True),
                ]
                add_running_datasets(running_datasets, s, h5py_k, h5py_new_handles, h5py_handles, used_datapoints)

                # Write resized images
                h5py_new_handles[s][s][h5py_k].create_dataset(
                    "image", (used_datapoints.shape[0], H, W, 3), dtype=np.uint8, compression=compression
                )

                for j, idx in enumerate(used_datapoints.tolist()):
                    data = h5py_handles[s][s][h5py_k]["image"][idx]
                    res = cv2.resize(data, (W, H))
                    h5py_new_handles[s][s][h5py_k]["image"][j] = res

            elif k == "pointclouds":
                used_pcds = np.stack(
                    [np.array(d[k][-(pcd_history):]) for d in dataset_config if d["sequence_key"] == s]
                )
                mapping = np.full(used_pcds.max() + 1, -1, dtype=np.int32)
                used_pcds = np.unique(used_pcds)
                mapping[used_pcds] = np.arange(used_pcds.shape[0])

                for j in range(len(dataset_config_new)):
                    if dataset_config_new[j]["sequence_key"] == s:
                        dataset_config_new[j][k] = mapping[np.array(dataset_config_new[j][k])[-(pcd_history):]].tolist()

                # Write messages other data
                running_datasets = [
                    ("header_frame_id", False),
                    ("tf_rotation_xyzw", True),
                    ("tf_translation", True),
                    ("header_stamp_secs", True),
                    ("header_stamp_nsecs", True),
                    ("header_seq", True),
                    ("intensity", True),
                    ("prob_ground", True),
                    ("prob_obstacle", True),
                    ("prob_trail", True),
                    ("prob_tree", True),
                    ("valid", True),
                    ("x", True),
                    ("y", True),
                    ("z", True),
                ]
                add_running_datasets(running_datasets, s, h5py_k, h5py_new_handles, h5py_handles, used_pcds)

            elif k == "gridmap_micro" or k == "gridmap_gt_micro":
                # overwrite the dataset config
                if k == "gridmap_micro":
                    for j in range(len(dataset_config_new)):
                        if dataset_config_new[j]["sequence_key"] == s:
                            dataset_config_new[j][k] = mapping[dataset_config_new[j][k]]

                # Write messages other data
                running_datasets = [
                    ("layers", False),
                    ("length", False),
                    ("resolution", False),
                    ("header_frame_id", False),
                    ("header_seq", True),
                    ("header_stamp_nsecs", True),
                    ("header_stamp_secs", True),
                    ("orientation_xyzw", True),
                    ("position", True),
                    ("tf_rotation_xyzw", True),
                    ("tf_translation", True),
                ]

                add_running_datasets(running_datasets, s, h5py_k, h5py_new_handles, h5py_handles, used_datapoints)

                shape = (used_datapoints.shape[0],) + h5py_handles[s][s][h5py_k]["data"].shape[1:]
                # Write gridmap data
                h5py_new_handles[s][s][h5py_k].create_dataset("data", shape, dtype="f", compression=compression)
                for j, idx in enumerate(used_datapoints.tolist()):
                    h5py_new_handles[s][s][h5py_k]["data"][j] = h5py_handles[s][s][h5py_k]["data"][idx]
            else:
                print(k, " compression not specified")

    file = join(root_dir, s)
    print(f"Execute for compression: h5repack -i {file}_new.h5py -o {file}_subsample_{subsample}.h5py")
    # os.system(f"h5repack -i {file}_new.h5py -o {file}_new_packed_subsample_{subsample}.h5py")

for j in range(len(dataset_config_new)):
    dataset_config_new[j]["sequence_key"] = dataset_config_new[j]["sequence_key"] + f"_subsample_{subsample}"

with open(join(root_dir, f"dataset_config_subsample_{subsample}.pkl"), "wb") as handle:
    pickle.dump(dataset_config_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
