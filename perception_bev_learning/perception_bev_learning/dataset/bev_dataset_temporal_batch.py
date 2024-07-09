from perception_bev_learning.utils import load_yaml, load_pkl, get_H, get_H_h5py
from perception_bev_learning.utils import (
    normalize_img,
    img_transform,
    inv,
    get_gravity_aligned,
)

import torch
from os.path import join
from PIL import Image
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import rotate, affine, center_crop
from dataclasses import asdict
from torch import from_numpy as fn
from torchvision.transforms.functional import pad, erase

import h5py
import random
from pytictac import ClassTimer, ClassContextTimer, cpu_accumulate_time, CpuTimer
import os
from typing import Dict, Any
from perception_bev_learning.dataset import BevDataset
from perception_bev_learning.dataset import find_continuous_sequences


class BevDatasetTemporalBatch(BevDataset):
    """
    Returns batched sequential data
    Dictionary of items with Tensor of shape B x N x Item for Tensors and N x B x Item for dictionaries (Pointcloud data)
    """

    def __init__(
        self, cfg: Dict[str, Any], mode: str = None, deterministic_shuffle=None
    ):
        self.cfg = cfg

        if mode is not None:
            self.mode = mode
        else:
            self.mode = str(self.cfg.mode)
        if deterministic_shuffle is not None:
            self.cfg.deterministic_shuffle = deterministic_shuffle

        self.image_keys = cfg.image_keys
        self.pointcloud_key = cfg.pointcloud_key
        self.gvom_key = cfg.gvom_key
        self.sequence_length = cfg.sequence_length

        self.dataset_config = load_pkl(join(cfg.root_dir, cfg.cfg_file))
        self.dataset_config = [d for d in self.dataset_config if d["mode"] == self.mode]

        # Get all sequences in the current dataset
        seq = [s["sequence_key"] for s in self.dataset_config]
        unique_sequences = np.unique(np.array(seq)).tolist()
        sequence_indices = {
            seq_name: np.where(np.array(seq) == seq_name)[0]
            for seq_name in unique_sequences
        }

        self.idx_sequences = []
        # Here it is assumed that the samples of the sequences are ordered
        for seq_name, indices in sequence_indices.items():
            continuous_sequences = find_continuous_sequences(
                indices, seq_length=self.sequence_length
            )
            self.idx_sequences.extend(continuous_sequences)

            if continuous_sequences:
                print(
                    f"Sequence: {seq_name}, Continuous Sequences: {continuous_sequences}"
                )

        print(f"Number of sequences are {len(self.idx_sequences)}")
        print(
            f"Opening in mode {self.mode} the following h5py files: ",
            unique_sequences,
        )
        print(f"Using {cfg.root_dir} to load the hdf5 files")
        self.h5py_handles = {
            s: h5py.File(join(cfg.root_dir, s + ".h5py"), "r") for s in seq
        }

        self.length = len(self.idx_sequences)
        self.setup_target_clip_scale()

        if self.cfg.deterministic_shuffle:
            random.shuffle(self.idx_sequences)

        self._cctk = 0
        self._cct = ClassTimer(
            objects=[self], names=["DataLoader"], enabled=cfg.enable_profiling
        )

    def __len__(self):
        return self.length

    @cpu_accumulate_time
    def __getitem__(self, index):
        """
        Returns a Dictionary of lists of N (=sequence length) items
        imgs_list -> N x 4 x 3 x H x W
        rots_list -> N x 4 x 3 x 3
        trans_list -> N x 4 x 3
        intrins_list -> N x 4 x 3 x 3
        post_rots_list -> N x 4 x 3 x 3
        post_trans_list -> N x 4 X 3
        target_list -> N x 2, 512, 512
        aux_list -> N x 6, 512, 512
        img_plots_list,
        index_list,
        gmr_list,
        H_sg__map_list,
        pcd_new_list,
        gvom_list
        """
        indices_new = self.idx_sequences[index]

        imgs_list = []
        rots_list = []
        trans_list = []
        intrins_list = []
        post_rots_list = []
        post_trans_list = []
        target_list = []
        aux_list = []
        gmr_list = []
        img_plots_list = []
        pcd_new_list = []
        gvom_list = []
        H_sg__map_list = []
        index_list = []

        for i in range(self.sequence_length):
            index_new = indices_new[i]
            datum = self.dataset_config[index_new]
            index_list.append(torch.Tensor(index_new))
            (
                target,
                aux,
                H_sensor_gravity__map,
                grid_map_resolution,
            ) = self.get_gridmap_data(datum)
            target_list.append(target)
            aux_list.append(aux)
            H_sg__map_list.append(H_sensor_gravity__map)
            gmr_list.append(grid_map_resolution)

            if self.cfg.return_image:
                (
                    imgs,
                    rots,
                    trans,
                    intrins,
                    post_rots,
                    post_trans,
                    img_plots,
                ) = self.get_image_data(datum, H_sensor_gravity__map)
            else:
                imgs, rots, trans, intrins, post_rots, post_trans, img_plots = tuple(
                    [torch.tensor([[[0]]])] * 7
                )

            imgs_list.append(imgs)
            rots_list.append(rots)
            trans_list.append(trans)
            intrins_list.append(intrins)
            post_rots_list.append(post_rots)
            post_trans_list.append(post_trans)
            img_plots_list.append(img_plots)

            if self.cfg.return_n_pointclouds != 0:
                pcd_new = self.get_raw_pcd_data(datum, H_sensor_gravity__map)

            else:
                pcd_new = {}

            if self.cfg.return_gvom_cloud:
                gvom = self.get_gvom_data(datum, H_sensor_gravity__map)
            else:
                gvom = {}

            pcd_new_list.append(pcd_new)
            gvom_list.append(gvom)

        if self.ct_enabled:
            print(self._cct)

        return {
            "imgs": imgs_list,
            "rots": rots_list,
            "trans": trans_list,
            "intrins": intrins_list,
            "post_rots": post_rots_list,
            "post_trans": post_trans_list,
            "target": target_list,
            "aux": aux_list,
            "img_plots": img_plots_list,
            "gm_res": gmr_list,
            "index": index_list,
            "H_sg_map": H_sg__map_list,
            "pcd": pcd_new_list,
            "gvom": gvom_list,
        }


def collate_fn_temporal(batch):
    """
    Return Tuple of
    N x B x Item if dictionary (e.g. Pointclouds)
    B x N x Item if tensors,
    where N is sequence length and B is batch size
    """
    output_batch = {}

    for key, values in batch[0].items():
        if type(values[0]) != dict:
            # iterate over tuple of data
            BN_data = torch.stack(
                [
                    torch.stack([item[key][n] for n in range(len(item[key]))])
                    for item in batch
                ]
            )
            output_batch[key] = BN_data
        else:
            # dicts are raw pointclouds
            res_list = []
            for n in range(len(batch[0][key])):
                res = {}
                stacked_scans_ls = []
                stacked_scan_indexes = []
                for j in range(len(batch)):
                    stacked_scans_ls.append(torch.cat(batch[j][key][n]["points"]))
                    stacked_scan_indexes.append(
                        torch.tensor(
                            [scan.shape[0] for scan in batch[j][key][n]["points"]]
                        )
                    )
                res["points"] = torch.cat(stacked_scans_ls)
                res["scan"] = torch.cat(stacked_scan_indexes)
                res["batch"] = torch.stack(stacked_scan_indexes).sum(1)
                res_list.append(res)

            output_batch[key] = res_list

    return output_batch

    # for i in range(len(batch[0])):
    #     if type(batch[0][i][0]) != dict:
    #         # iterate over tuple of data
    #         BN_data = torch.stack(
    #             [
    #                 torch.stack([item[i][n] for n in range(len(item[i]))])
    #                 for item in batch
    #             ]
    #         )
    #         output_batch.append(BN_data)
    #     else:
    #         # dicts are raw pointclouds
    #         res_list = []
    #         for n in range(len(batch[0][i])):
    #             res = {}
    #             stacked_scans_ls = []
    #             stacked_scan_indexes = []
    #             for j in range(len(batch)):
    #                 stacked_scans_ls.append(torch.cat(batch[j][i][n]["points"]))
    #                 stacked_scan_indexes.append(
    #                     torch.tensor(
    #                         [scan.shape[0] for scan in batch[j][i][n]["points"]]
    #                     )
    #                 )
    #             res["points"] = torch.cat(stacked_scans_ls)
    #             res["scan"] = torch.cat(stacked_scan_indexes)
    #             res["batch"] = torch.stack(stacked_scan_indexes).sum(1)
    #             res_list.append(res)

    #         output_batch.append(res_list)

    # return tuple(output_batch)
