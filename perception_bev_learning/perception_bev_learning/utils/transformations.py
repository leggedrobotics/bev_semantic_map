import torch
import numpy as np
from pyquaternion import Quaternion  # w, x, y, z
from scipy.spatial.transform import Rotation as R
from torch import from_numpy as fn


def get_rot(q_ros):
    return torch.from_numpy(
        Quaternion([q_ros[3], q_ros[0], q_ros[1], q_ros[2]]).rotation_matrix
    ).type(torch.float64)


def get_H_h5py(t, q):
    H = torch.eye(4)
    H[:3, 3] = torch.tensor(t).type(torch.float32)
    H[:3, :3] = get_rot(q)
    return H


def get_H(tf, offset=4):
    H = torch.eye(4)
    H[:3, 3] = torch.tensor(tf[offset]).type(torch.float32)
    H[:3, :3] = get_rot(tf[offset + 1])
    return H


def inv(H):
    H_ = H.clone()
    H_[:3, :3] = H.T[:3, :3]
    H_[:3, 3] = -H.T[:3, :3] @ H[:3, 3]
    return H_


def get_gravity_aligned(H_f__map):
    ypr = R.from_matrix(H_f__map.clone().cpu().numpy()[:3, :3]).as_euler(
        seq="zyx", degrees=True
    )
    H_g__map = H_f__map.clone()
    H_delta = torch.eye(4)

    ypr[0] = 0
    H_delta[:3, :3] = fn(R.from_euler(seq="zyx", angles=ypr, degrees=True).as_matrix())
    H_g__map = inv(H_delta) @ H_g__map

    return H_g__map


def invert_se3(T):
    T_inv = torch.zeros_like(T)
    T_inv[3, 3] = 1.0
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv
