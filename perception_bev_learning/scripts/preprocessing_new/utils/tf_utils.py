import numpy as np

from pyquaternion import Quaternion  # w, x, y, z
from scipy.spatial.transform import Rotation as R
import torch
from torch import from_numpy as fn
from collections import deque

def get_np_rot(q_ros):
    return np.array(
        Quaternion([q_ros[3], q_ros[0], q_ros[1], q_ros[2]]).rotation_matrix
    )


def get_np_T(t, q):
    H = np.eye(4)
    H[:3, 3] = np.array(t)
    H[:3, :3] = get_np_rot(q)
    return H

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

def get_yaw_oriented_sg(H_map__sg):

    H_sg__map = inv(H_map__sg) 
    ypr = R.from_matrix(H_sg__map.clone().cpu().numpy()[:3, :3]).as_euler(
        seq="zyx", degrees=True
    )
    H_sgyaw__map = H_sg__map.clone()
    H_delta = torch.eye(4)

    ypr[1] = 0
    ypr[2] = 0
    H_delta[:3, :3] = fn(R.from_euler(seq="zyx", angles=ypr, degrees=True).as_matrix())
    H_sgyaw__map = inv(H_delta) @ H_sgyaw__map

    return inv(H_sgyaw__map)

def connected_component_search(gridmap, threshold=2.0):
    rows, cols = gridmap.shape
    center = (rows // 2, cols // 2)
    visited = np.zeros_like(gridmap, dtype=bool)
    component = np.full_like(gridmap, np.nan)
    
    # Directions for the 4-connected neighbors (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r, c, current_height):
        return (0 <= r < rows and 0 <= c < cols and
                not visited[r, c] and
                not np.isnan(gridmap[r, c]) and
                abs(gridmap[r, c] - current_height) <= threshold)
    
    # Initialize BFS
    queue = deque([center])
    visited[center] = True
    component[center] = gridmap[center]
    
    while queue:
        r, c = queue.popleft()
        current_height = gridmap[r, c]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc, current_height):
                visited[nr, nc] = True
                component[nr, nc] = gridmap[nr, nc]
                queue.append((nr, nc))
    
    return component
