import h5py
import argparse
from pathlib import Path
from perception_bev_learning.utils import load_pkl, load_env
import pickle
from perception_bev_learning.cfg import ExperimentParams
from os.path import join
import numpy as np

if __name__ == "__main__":
    env = load_env()
    folder = "/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/add_supervision/"
    data = load_pkl(folder + "dataset_config_clean_seperation_subsample_1.pkl")

    print("Training samples", len([d for d in data if d["mode"] == "train"]))
    print("Testing samples", len([d for d in data if d["mode"] == "test"]))
    print("Validation samples", len([d for d in data if d["mode"] == "val"]))

    seq = [s["sequence_key"] for s in data]
    seq = np.unique(np.array(seq)).tolist()
    h5py_handles = {s: h5py.File(join(env["dataset_root_dir"], s + ".h5py"), "r") for s in seq}

    xy = None
    for m in ["train", "test", "val"]:
        total_distance = 0
        total_time = 0
        for s in seq:
            reduced_dataset = [d for d in data if d["sequence_key"] == s and d["mode"] == m]
            xy = None
            total_dis = 0
            total_t = 0
            for _d in reduced_dataset:
                f = h5py_handles[_d["sequence_key"]][_d["sequence_key"].replace("_with_data", "")]
                img = f["crl_rzr_multisense_front_aux_semantic_image_rect_color_compressed"]
                idx = _d["image_front"]
                c_time = img["header_stamp_secs"][idx] + img["header_stamp_nsecs"][idx] * 10 ** (-9)
                if xy is not None:
                    total_dis += np.linalg.norm(xy - img["tf_translation_map__base_link"][idx][:2])
                    total_t += c_time - time

                time = c_time
                xy = img["tf_translation_map__base_link"][idx][:2]

            total_distance += total_dis
            total_time += total_t
            print(s, m, f" Distance {total_dis} Total Time {total_t}")

        print("TOTAL: ", m + f" Distance {total_distance} Total Time {total_time/60} Minutes")
