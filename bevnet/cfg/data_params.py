from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch
import os


@dataclass
class DataParams:
    dataset: str = "bevnet"
    mode: str = "train"

    # Sensors
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1

    # Image
    img_width: int = 640  # 640, 128; 720
    img_height: int = 480  # 480, 128; 540

    # Camera parameters
    intrin = [255.8245, 0.0000, 331.4361, 0.0000, 257.1399, 230.8981, 0.0000, 0.0000, 1.0000]  # 640 x 480
    trans_base_cam = [-1.1102230246251565e-16, 0.020499999999999907, -0.40449]
    rot_base_cam = [-0.5, 0.4999999999999999, -0.5, -0.5000000000000001]

    # Output settings
    target_shape: Tuple[int, int, int] = (1, 64, 64)
    aux_shape: Tuple[int, int, int] = (1, 64, 64)
    grid_map_resolution: float = 0.1

    def __post_init__(self):
        self.data_dir = os.path.join(f"/home/rschmid/RosBags/{self.dataset}", self.mode)


data: DataParams = DataParams()
