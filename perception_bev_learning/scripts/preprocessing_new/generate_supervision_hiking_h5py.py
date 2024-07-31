import sys
import argparse
import rospy
import numpy as np
from tqdm import tqdm
import cv2
from os.path import join, splitext

import torch
from torchvision.transforms.functional import affine
from tqdm import tqdm
import h5py
import copy
import signal
import sys
from torchvision.transforms.functional import pad
from torch.nn.functional import interpolate

from utils.h5py_writer import DatasetWriter
from utils.loading import load_pkl
from perception_bev_learning.ros import SimpleNumpyToRviz

from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pandas as pd
import pickle


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


class GenerateGTSupervision:
    def __init__(self, cfg):
        self.cfg = OmegaConf.load(cfg)
        OmegaConf.resolve(self.cfg)
        self.cfg = instantiate(self.cfg)

        self.visu = self.cfg.visualize
        self.override = self.cfg.override
        self.write = self.cfg.write
        # self.gridmap_size = self.cfg.HW_ext

        if self.visu:
            self.vis_pred = SimpleNumpyToRviz()
            self.vis_gt = SimpleNumpyToRviz(init_node=False, postfix="_gt")
            rospy.on_shutdown(self.shutdown_callback)

        # Dictionary for all gridmap objects to be converted
        self.gridmap_cfg_dict = {
            gridmap.obs_name: gridmap for gridmap in self.cfg.dataset.values()
        }
        self.gridmap_fusion_dict = {
            gridmap.obs_name: gridmap.fusions for gridmap in self.cfg.dataset.values()
        }
        self.obs_key_to_gt_key = {
            gridmap.obs_name: gridmap.gt_obs_name
            for gridmap in self.cfg.dataset.values()
        }

        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self):
        try:
            self.file.close()
        except:
            pass
        sys.exit()

    def compute(self, h5py_file, pkl_file):
        self.file = h5py.File(h5py_file, "r+")
        self.pd_df = pd.DataFrame(load_pkl(pkl_file))

        # Create gt_key colums if they don't exist
        headers = self.pd_df.columns.tolist()
        for gt_key in self.obs_key_to_gt_key.values():
            if gt_key not in headers:
                self.pd_df[gt_key] = np.zeros(self.pd_df.shape[0], dtype=int)

        if len(self.file.keys()) > 1:
            # Multiple can be processed but for now it is expected that H5py are separate for each seq
            print("Multiple sequences present in h5py file. Not processing further")
            exit()

        seq_name = list(self.file.keys())[0]
        h5py_seq = self.file[seq_name]

        # Extract pickle config file for h5py seq
        df = self.pd_df[self.pd_df["sequence_key"] == seq_name]
        df_len = df.shape[0]
        # idx_mapping = np.arange(0, df_len)
        idx_mapping = df.index.tolist()

        # Set the correct config indices
        for gt_key in self.obs_key_to_gt_key.values():
            self.pd_df.loc[df.index, gt_key] = np.arange(0, df_len)

        if self.write:
            dataset_writer = DatasetWriter(h5py_file, open_file=self.file)

        for gridmap_key, fusions in self.gridmap_fusion_dict.items():
            gridmap_gt_key = self.obs_key_to_gt_key[gridmap_key]
            max_acc_time = self.gridmap_cfg_dict[gridmap_key].max_acc_time


            if self.override:
                try:
                    del h5py_seq[gridmap_gt_key]
                except:
                    print("GT does not exist")
                    pass
            else:
                if gridmap_gt_key in h5py_seq.keys():
                    N1 = df_len
                    N2 = h5py_seq[gridmap_gt_key]["header_seq"].shape[0]
                    if N1 == N2:
                        print(
                            "Did not further process sequence given that GT already exists!"
                        )
                        continue

            h5py_gridmap = h5py_seq[gridmap_key]
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

            length = np.array(h5py_gridmap["length"])
            res = np.array(h5py_gridmap["resolution"])

            layers = [l.decode("utf-8") for l in h5py_gridmap["layers"]]
            # layers += ["robust_reliable"]
            # # layers += ["elevation_est"]

            layer_idx_dict = {l: layers.index(l) for l in layers}

            N, C, H, W = h5py_gridmap["data"].shape
            # print(N, C, H, W)
            H_ext, W_ext = self.gridmap_cfg_dict[gridmap_key].HW_ext
            pad_val = [int((H_ext - H) / 2), int((W_ext - W) / 2)]

            possible_idx = np.arange(N)

            if self.gridmap_cfg_dict[gridmap_key].update_with_micro_range:
                micro_range_key = self.gridmap_cfg_dict[gridmap_key].micro_obs_name
                h5py_gridmap_micro = h5py_seq[micro_range_key]
                t = get_time(h5py_gridmap_micro)

                # Ensure data is sorted according to timestamp
                t_ls = t[:, 0].tolist()
                t_sort = copy.deepcopy(t_ls)
                t_sort.sort()
                assert (
                    t_ls == t_sort
                ), "GridMap data is not sorted correctly according to timestamp"

                pose_micro = np.array(h5py_gridmap_micro["position"])
                timestamps_micro = np.array(t[:, 0])

                length_micro = np.array(h5py_gridmap_micro["length"])
                res_micro = np.array(h5py_gridmap_micro["resolution"])

                N_m = h5py_gridmap_micro["data"].shape[0]

                possible_idx_micro = np.arange(N_m)

            ####
            # interested_idx = np.arange(280, 290)
            interested_idx = idx_mapping
            ####
            try:
                with tqdm(
                    total=len(idx_mapping),
                    desc="Total",
                    colour="green",
                    position=1,
                    bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
                ) as pbar:
                    for idx in idx_mapping:  
                        gridmap_idx = df.at[idx, gridmap_key]

                        if idx in interested_idx:
                            if not self.override:
                                if gridmap_gt_key in h5py_seq.keys():
                                    if (
                                        h5py_seq[gridmap_gt_key]["header_seq"].shape[0]
                                        > idx
                                    ):
                                        if bool(
                                            h5py_seq[gridmap_gt_key]["header_seq"][idx]
                                            == h5py_seq[gridmap_key]["header_seq"][
                                                gridmap_idx
                                            ]
                                        ):
                                            pbar.update(1)
                                            print("already processed continue")
                                            continue

                            m_pose = (
                                np.linalg.norm(pose - pose[gridmap_idx], ord=1, axis=1)
                                < length[0]
                            )
                            m_time = (timestamps - timestamps[gridmap_idx]) < max_acc_time
                            m = m_pose * m_time

                            # Increase the GT Map Size to root 2 times
                            # map1 = np.zeros((C, H_ext, W_ext), dtype=np.float32)
                            map1 = np.full((C, H_ext, W_ext), np.nan, dtype=np.float32)
                            map2 = np.zeros((C, H_ext, W_ext), dtype=np.float32)
                            H_c = int(H_ext / 2)
                            W_c = int(W_ext / 2)

                            reference_pose = pose[gridmap_idx]
                            subdiv = max(1, int(possible_idx[m].shape[0] / 100))

                            # First we do the fusion of the current gridmap key

                            for k in possible_idx[m][::subdiv]:
                                data = np.array(h5py_gridmap["data"][k])
                                data = torch.from_numpy(data)[None]

                                # Pad the data according to the ext
                                data = pad(data, pad_val, torch.nan, "constant")

                                shift = -(reference_pose - pose[k]) / res
                                sh = [shift[1], shift[0]]

                                data_shifted = affine(
                                    data,
                                    angle=0,
                                    translate=sh,
                                    scale=1,
                                    shear=0,
                                    center=(H_c, W_c),
                                    fill=torch.nan,
                                )[0].numpy()

                                for j, l in enumerate(layers):
                                    map1[j], map2[j] = fusions[l].fuse(
                                        data_shifted[j], map1[j], map2[j], np.ones_like(data_shifted[j])
                                    )

                                if self.visu:
                                    shrink = 1
                                    self.vis_pred.gridmap_arr(
                                        data_shifted[:, shrink:-shrink, shrink:-shrink],
                                        res=res,
                                        layers=layers,
                                    )

                            out = map1.copy()
                            for j, l in enumerate(layers):
                                prediction_invalid = map2 == 0
                                map1[prediction_invalid] = np.nan

                                m = ~np.isnan(map1)

                                if m.sum() == 0:
                                    out[m] = np.nan
                                else:
                                    out[m] = map1[m] / map2[m]

                                out[prediction_invalid] = np.nan

                            # # TODO: Add the micro map Fusion

                            # ####################################################
                            # print(pose_micro.shape, pose[gridmap_idx].shape)
                            # print(length[0]/2 + length_micro[0])
                            m_pose = (
                                np.linalg.norm(pose_micro - pose[gridmap_idx], ord=1, axis=1)
                                < (1.41 * length[0]/2 + length_micro[0])
                            )
                            m_time = (timestamps_micro - timestamps[gridmap_idx]) < max_acc_time
                            m = m_pose * m_time

                            # Increase the GT Map Size to root 2 times
                            # map1 = np.zeros((C, H_ext, W_ext), dtype=np.float32)
                            map1 = np.full((C, H_ext, W_ext), np.nan)
                            map2 = np.zeros((C, H_ext, W_ext), dtype=np.float32)
                            H_c = int(H_ext / 2)
                            W_c = int(W_ext / 2)

                            reference_pose = pose[gridmap_idx]
                            subdiv = max(1, int(possible_idx_micro[m].shape[0] / 100))

                            for k in possible_idx_micro[m][::subdiv]:
                                data = np.array(h5py_gridmap_micro["data"][k])
                                data = torch.from_numpy(data)[None]

                                # Downsample the micro range map
                                data = interpolate(data, scale_factor=res_micro[0]/res[0])
                                _, _, H_m, W_m = data.shape

                                pad_val_micro = [int((H_ext - H_m) / 2), int((W_ext - W_m) / 2)]

                                # Pad the data according to the ext TODO
                                data = pad(data, pad_val_micro, torch.nan, "constant")

                                shift = -(reference_pose - pose_micro[k]) / res
                                sh = [shift[1], shift[0]]

                                data_shifted = affine(
                                    data,
                                    angle=0,
                                    translate=sh,
                                    scale=1,
                                    shear=0,
                                    center=(H_c, W_c),
                                    fill=torch.nan,
                                )[0].numpy()

                                for j, l in enumerate(layers):
                                    map1[j], map2[j] = fusions[l].fuse(
                                        data_shifted[j], map1[j], map2[j], np.ones_like(data_shifted[j])
                                    )

                                if self.visu:
                                    shrink = 1
                                    self.vis_pred.gridmap_arr(
                                        data_shifted[:, shrink:-shrink, shrink:-shrink],
                                        res=res,
                                        layers=layers,
                                    )

                            out_micro = map1.copy()
                            for j, l in enumerate(layers):
                                prediction_invalid = map2 == 0
                                map1[prediction_invalid] = np.nan

                                m = ~np.isnan(map1)

                                if m.sum() == 0:
                                    out_micro[m] = np.nan
                                else:
                                    out_micro[m] = map1[m] / map2[m]

                                out_micro[prediction_invalid] = np.nan
                            
                            ###################################################

                            # Merge both maps
                            micro_valid = ~np.isnan(out_micro)
                            
                            out[micro_valid] = out_micro[micro_valid]

                            ###################################################


                            ######## Add the current Micro Map ################
                            m = (
                                np.linalg.norm(pose_micro - pose[gridmap_idx], ord=1, axis=1)
                                < 1
                            )

                            map1 = np.full((C, H_ext, W_ext), np.nan)
                            map2 = np.zeros((C, H_ext, W_ext), dtype=np.float32)
                            H_c = int(H_ext / 2)
                            W_c = int(W_ext / 2)

                            reference_pose = pose[gridmap_idx]
                            subdiv = max(1, int(possible_idx_micro[m].shape[0] / 100))

                            for k in possible_idx_micro[m][::subdiv]:
                                data = np.array(h5py_gridmap_micro["data"][k])
                                data = torch.from_numpy(data)[None]

                                # Downsample the micro range map
                                data = interpolate(data, scale_factor=res_micro[0]/res[0])
                                data = data[..., 9:-9, 9:-9] # Hack for removing the boundary un preferred

                                _, _, H_m, W_m = data.shape

                                pad_val_micro = [int((H_ext - H_m) / 2), int((W_ext - W_m) / 2)]

                                # Pad the data according to the ext TODO
                                data = pad(data, pad_val_micro, torch.nan, "constant")

                                shift = -(reference_pose - pose_micro[k]) / res
                                sh = [shift[1], shift[0]]

                                data_shifted = affine(
                                    data,
                                    angle=0,
                                    translate=sh,
                                    scale=1,
                                    shear=0,
                                    center=(H_c, W_c),
                                    fill=torch.nan,
                                )[0].numpy()

                                for j, l in enumerate(layers):
                                    map1[j], map2[j] = fusions[l].fuse(
                                        data_shifted[j], map1[j], map2[j], np.ones_like(data_shifted[j])
                                    )

                                if self.visu:
                                    shrink = 1
                                    self.vis_pred.gridmap_arr(
                                        data_shifted[:, shrink:-shrink, shrink:-shrink],
                                        res=res,
                                        layers=layers,
                                    )

                            out_micro_curr = map1.copy()
                            for j, l in enumerate(layers):
                                prediction_invalid = map2 == 0
                                map1[prediction_invalid] = np.nan

                                m = ~np.isnan(map1)

                                if m.sum() == 0:
                                    out_micro_curr[m] = np.nan
                                else:
                                    out_micro_curr[m] = map1[m] / map2[m]

                                out_micro_curr[prediction_invalid] = np.nan
                            
                            ###################################################

                            # Merge both maps
                            micro_valid = ~np.isnan(out_micro_curr)
                            out[micro_valid] = out_micro_curr[micro_valid]
                        
                        else:
                            out = np.zeros((C, H_ext, W_ext), dtype=np.float32)


                        static = {
                            "layers": layers,
                            "resolution": np.array(h5py_gridmap["resolution"]),
                            "length": np.array((res * H_ext, res * W_ext)).reshape(-1),
                            "header_frame_id": "crl_rzr/map",
                        }
                        dynamic = {
                            "data": out,
                            "header_seq": h5py_gridmap["header_seq"][gridmap_idx],
                            "header_stamp_nsecs": h5py_gridmap["header_stamp_nsecs"][
                                gridmap_idx
                            ],
                            "header_stamp_secs": h5py_gridmap["header_stamp_secs"][
                                gridmap_idx
                            ],
                            "orientation_xyzw": h5py_gridmap["orientation_xyzw"][
                                gridmap_idx
                            ],
                            "position": h5py_gridmap["position"][gridmap_idx],
                            "tf_rotation_xyzw": h5py_gridmap["tf_rotation_xyzw"][
                                gridmap_idx
                            ],
                            "tf_translation": h5py_gridmap["tf_translation"][
                                gridmap_idx
                            ],
                        }
                        if self.write:
                            dataset_writer.add_static(
                                sequence=seq_name,
                                fieldname=gridmap_gt_key,
                                data_dict=static,
                            )
                            dataset_writer.add_data(
                                sequence=seq_name,
                                fieldname=gridmap_gt_key,
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
    parser = argparse.ArgumentParser(description="Add GT Fused Gridmaps")
    parser.add_argument("--cfg", type=str, help="path to config yaml file ")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # Get the h5py files
    if Path(cfg.h5_files).is_dir():
        h5py_files = [str(s) for s in Path(cfg.h5_files).rglob("*.h5py")]
        h5py_files.sort(reverse=True)
    elif splitext(cfg.h5_files)[-1] == ".h5py":
        h5py_files = [cfg.h5_files]
    else:
        print("Correct path for h5py files is not provided in cfg")
        exit()

    print(f"The following h5py files will be processed: {h5py_files}")

    for f in h5py_files:
        print("Processing file: ", f)
        pkl_cfg_file = splitext(f)[0] + ".pkl"
        print("Config file is: ", pkl_cfg_file)
        gt_sup = GenerateGTSupervision(cfg=args.cfg)
        gt_sup.compute(h5py_file=f, pkl_file=pkl_cfg_file)

        # Save the Pandas DF
        list_of_dicts = gt_sup.pd_df.to_dict(orient="records")
        with open(pkl_cfg_file, "wb") as handle:
            pickle.dump(list_of_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished processing all files")
