from abc import ABC, abstractmethod
from typing import Any
from typing import Tuple, Dict, List, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch


def voxelize_pcd_scans(
    x: torch.Tensor, batch: torch.Tensor, scan: torch.Tensor, gm_dim: List[int], gm_res: List[float]
):
    # Verified this is now the exact same code used in the dataloader

    # simple voxelization
    gm_dim = torch.tensor(gm_dim, device=x.device)
    gm_res = torch.tensor(gm_res, device=x.device)
    x += (gm_dim * gm_res) / 2
    # compute voxel indices
    index = (x / gm_res).type(torch.long)

    # mask indices
    m = (
        (index[:, 0] >= 0)
        * (index[:, 1] >= 0)
        * (index[:, 2] >= 0)
        * (index[:, 0] < gm_dim[0])
        * (index[:, 1] < gm_dim[1])
        * (index[:, 2] < gm_dim[2])
    )

    # fill voxel volume
    BS = len(batch)
    voxel_volume = torch.zeros((BS, 1) + tuple(gm_dim), device=x.device)

    start = 0
    for i in range(BS):
        bi = index[start : start + int(batch[i])]
        valid = m[start : start + int(batch[i])]
        voxel_volume[i, 0, bi[valid, 0], bi[valid, 1], bi[valid, 2]] = 1.0
        start += int(batch[i])

    return voxel_volume.permute(0, 1, 4, 2, 3)  # B, S, C, H, W