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


def find_continuous_sequences(indices, seq_length):
    continuous_sequences = []
    current_sequence = []

    for idx in indices:
        if not current_sequence or idx == current_sequence[-1] + 1:
            current_sequence.append(idx)
        else:
            current_sequence = [idx]

        if len(current_sequence) == seq_length:
            continuous_sequences.append(current_sequence.copy())
            current_sequence = []

    return continuous_sequences


class BevDatasetTemporal(BevDataset):
    """
    Dataset class for loading temporal data in sequences
    This should be used with a proper batch sampler so that the returned samples are consective upto specified sequence length
    self.idx_sequences is a list of lists where each list is a sequence of indices of length self.sequence_length
    self.idx_sequences will be used by the batch sampler to return consecutive samples
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

        self.length = len(self.dataset_config)
        self.setup_target_clip_scale()
        self.index_mapping = np.arange(0, self.length).astype(np.int32).tolist()

        if self.cfg.deterministic_shuffle:
            random.shuffle(self.idx_sequences)

        self._cctk = 0
        self._cct = ClassTimer(
            objects=[self], names=["DataLoader"], enabled=cfg.enable_profiling
        )

    def __len__(self):
        return self.index_mapping

    @cpu_accumulate_time
    def __getitem__(self, index):
        """
        Returns a dictionary of items
        imgs -> 4 x 3 x H x W
        rots -> 4 x 3 x 3
        trans-> 4 x 3
        intrins -> 4 x 3 x 3
        post_rots -> 4 x 3 x 3
        post_trans -> 4 X 3
        target -> 2, 512, 512
        aux -> 6, 512, 512
        img_plots,
        gridmap_resolution,
        H_sg__map,
        pcd_new,
        gvom
        """

        index_new = self.index_mapping[index]
        datum = self.dataset_config[index_new]

        target, aux, H_sensor_gravity__map, grid_map_resolution = self.get_gridmap_data(
            datum
        )

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

        if self.cfg.return_n_pointclouds != 0:
            pcd_new = self.get_raw_pcd_data(datum, H_sensor_gravity__map)

        else:
            pcd_new = {}

        if self.cfg.return_gvom_cloud:
            gvom = self.get_gvom_data(datum, H_sensor_gravity__map)
        else:
            gvom = {}

        if self.ct_enabled:
            print(self._cct)

        return {
            "imgs": imgs,
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "post_rots": post_rots,
            "post_trans": post_trans,
            "target": target,
            "aux": aux,
            "img_plots": img_plots,
            "gm_res": grid_map_resolution,
            "index": torch.tensor(index),
            "H_sg_map": H_sensor_gravity__map,
            "pcd": pcd_new,
            "gvom": gvom,
        }
