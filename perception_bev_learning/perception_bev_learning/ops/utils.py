from abc import ABC, abstractmethod
from typing import Any
from typing import Tuple, Dict, List, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch


def voxelize_pcd_scans(
    x: torch.Tensor,
    batch: torch.Tensor,
    scan: torch.Tensor,
    gm_dim: List[int],
    gm_res: List[float],
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


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    TODO Check copyright
    Adapted from https://github.com/qq456cvb/Point-Transformers/blob/master/models/Menghao/model.py
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(ref, query):
    """return indices of ref for each query point. L2 norm
    Args:
        ref ([type]): points * 3
        query ([type]): tar_points * 3
    Returns:
        [knn]: distance = query * 1 , indices = query * 1
    """
    min_val = torch.min(torch.cat([ref, query], dim=0), 0)[0]
    max_val = torch.max(torch.cat([ref, query], dim=0), 0)[0]

    bin_res = torch.tensor((5, 5, 5), device=ref.device, dtype=ref.dtype)
    max_bins = ((max_val - min_val) / bin_res).type(torch.long)
    unique_idx_mult = torch.tensor(
        (1, max_bins[0], max_bins[0] * max_bins[1]), device=ref.device, dtype=ref.dtype
    )
    idx_ref = ((ref - min_val) / bin_res).type(torch.long)
    idx_query = ((query - min_val) / bin_res).type(torch.long)

    unique_idx_ref = (idx_ref * unique_idx_mult).sum(dim=1)
    unique_idx_query = (idx_query * unique_idx_mult).sum(dim=1)
    torch.unique(unique_idx_ref, return_inverse=True, return_counts=True)
    # TODO use not the unquie idex and maybe unique to create the topk lookup thing between all points; finding a smart way to pad it may be sufficient for this operation; do a kNN for each index

    from pytorch3d.ops import ball_query

    # Try PyTorch 3D to do the thing
    res = ball_query(
        ref[None, :, :].type(torch.float32), query[None, :, :].type(torch.float32)
    )

    mp2 = ref[None, :20, :].repeat(query.shape[0], 1, 1)
    tp2 = query[:, None, :].repeat(1, ref.shape[0], 1)
    dist = torch.norm(mp2 - tp2, dim=2, p=None)
    print(dist.topk(1, largest=False)[0])

    dist_test = dist.repeat(2, 1, 1)
    dist_test[0, 0, :] += 1000
    dist_test.topk(1, largest=False)[0][0, 0]

    dist_test[0, 0, :999] += 1000
    knn = dist.topk(1, largest=False)

    return {"distance": knn[0], "idx": knn[1]}


if __name__ == "__main__":
    import pickle

    with open("tmp.pickle", "rb") as handle:
        a = pickle.load(handle)

    frustum_points = a["frustum_points"]
    lidar_points = a["lidar_points"]
    from pytorch3d.ops import ball_query

    res = torch.unique(
        ball_query(
            frustum_points[None, :, :].type(torch.float32),
            lidar_points[None, :400, :].type(torch.float32),
            radius=10,
        )[1]
    )[1:]

    ball_query(
        frustum_points[None, :, :].type(torch.float32),
        lidar_points[None, :, :].type(torch.float32),
    )

    print("start")
    knn(frustum_points, lidar_points)

    print("done")
    print("done")
