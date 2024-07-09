import h5py
import argparse
from pathlib import Path
from perception_bev_learning.utils import load_pkl
import pickle

if __name__ == "__main__":
    folder = "/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/add_supervision/"
    for sub in [1, 2, 5, 10, 20, 50, 100, 200]:
        for split_name in ["clean_seperation", "mixed_seperation"]:
            data = load_pkl(folder + "dataset_config.pkl")
            if split_name == "clean_seperation":
                test = []
                # CLEAR SPLIT SETUP LEFT Trajectories
                # red
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t9_top_of_hill_Wed_Jun__8_00-13-40_2022_utc_with_data"
                )
                # yellow
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t5_open_Tue_Jun__7_23-17-53_2022_utc_with_data"
                )
                # black
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t8_delta2_Tue_Jun__7_23-45-36_2022_utc_with_data"
                )
                # magenta
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t7_delta1_Tue_Jun__7_23-35-19_2022_utc_with_data"
                )

                # Right Trajectories
                train = []
                # green
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_C2_to_C1_jose_driving_Tue_Jun__7_17-41-33_2022_utc_with_data"
                )
                # orange
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_C1_to_B2_Tue_Jun__7_18-21-54_2022_utc_with_data"
                )
                # purple
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_B2_to_A4_Tue_Jun__7_19-10-17_2022_utc_with_data"
                )
                # cayne
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_A5_to_A4_Tue_Jun__7_19-21-15_2022_utc_with_data"
                )

            if split_name == "mixed_seperation":
                test = []
                # magenta
                test.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t7_delta1_Tue_Jun__7_23-35-19_2022_utc_with_data"
                )
                # cayne
                test.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_A5_to_A4_Tue_Jun__7_19-21-15_2022_utc_with_data"
                )

                train = []
                # green
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_C2_to_C1_jose_driving_Tue_Jun__7_17-41-33_2022_utc_with_data"
                )
                # orange
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_C1_to_B2_Tue_Jun__7_18-21-54_2022_utc_with_data"
                )
                # purple
                train.append(
                    "jpl6_camp_roberts_shakeout_y1_d2_t1_B2_to_A4_Tue_Jun__7_19-10-17_2022_utc_with_data"
                )
                # red
                train.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t9_top_of_hill_Wed_Jun__8_00-13-40_2022_utc_with_data"
                )
                # yellow
                train.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t5_open_Tue_Jun__7_23-17-53_2022_utc_with_data"
                )
                # black
                train.append(
                    "jpl6_camp_roberts_shakeout_y6_d2_t8_delta2_Tue_Jun__7_23-45-36_2022_utc_with_data"
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
