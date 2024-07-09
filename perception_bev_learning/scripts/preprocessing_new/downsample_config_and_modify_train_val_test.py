import h5py
import argparse
from pathlib import Path
from perception_bev_learning.utils import load_pkl
import pickle

if __name__ == "__main__":
    folder = "/data_large/manthan/camp_roberts_f/config/"
    for sub in [1, 2, 5, 10]:
        for split_name in ["clean_seperation"]:
            data = load_pkl(folder + "dataset_config.pkl")
            if split_name == "clean_seperation":
                test = []
                # CLEAR SPLIT SETUP LEFT Trajectories
                # red
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t9_top_of_hill_Wed_Jun__8_00-13-40_2022_utc"
                )
                test.append(
                    "jpl6_camp_roberts_y1_d3_t11_jeffery_create_trail_Wed_Jun__8_22-38-35_2022_utc"
                )
                test.append(
                    "jpl6_camp_roberts_y1_d3_t12_jeffery_delta2_Wed_Jun__8_22-48-12_2022_utc"
                )
                test.append(
                    "jpl6_camp_roberts_y1_d3_t13_jose_delta_trail_Wed_Jun__8_23-10-41_2022_utc"
                )
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t5_open_Tue_Jun__7_23-17-53_2022_utc"
                )
                test.append(
                    "jpl6_camp_roberts_y1_d3_t1_rivercross_metal_area_Wed_Jun__8_18-21-19_2022_utc"
                )

                # Right Trajectories
                train = []
                # green
                train.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t8_delta2_Tue_Jun__7_23-45-36_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t7_delta1_Tue_Jun__7_23-35-19_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t1_david_low_hill_Wed_Jun__8_18-37-00_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t1_hidden_obstacle_and_grass_Wed_Jun__8_18-29-02_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t1_rivercross_aggressive_Wed_Jun__8_18-03-46_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t1_rivercross_Wed_Jun__8_17-32-46_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t10_jeffery_high_speed_trails_Wed_Jun__8_22-22-37_2022_utc"
                )
                train.append(
                    "jpl6_camp_roberts_y1_d3_t13_jose_steep_hill_Wed_Jun__8_22-59-26_2022_utc"
                )

            for i in range(len(data)):
                if data[i]["sequence_key"] in test:
                    data[i]["mode"] = "test"

            for t in train:
                el = []
                nrs = [
                    j for j, sample in enumerate(data) if sample["sequence_key"] == t
                ]
                threshold = int(len(nrs) * 0.8)
                for n in nrs[:threshold]:
                    data[n]["mode"] = "train"
                for n in nrs[threshold:]:
                    data[n]["mode"] = "val"

            with open(
                folder + f"dataset_config_{split_name}_subsample_{sub}.pkl", "wb"
            ) as file:
                pickle.dump(data[::sub], file, protocol=pickle.HIGHEST_PROTOCOL)
            print("done")
