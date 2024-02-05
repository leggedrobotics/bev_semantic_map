from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch
import os


@dataclass
class DataParams:
    dataset: str = "bevnet2"
    # dataset: str = "single_data"
    mode: str = "train"
    data_dir_base: str = "/home/rschmid/RosBags"

    # Sensors
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1

    # Image
    img_width: int = 720  # 640, 128; 720
    img_height: int = 540  # 480, 128; 540

    # Camera parameters
    # intrin = [255.8245, 0.0000, 331.4361, 0.0000, 257.1399, 230.8981, 0.0000, 0.0000, 1.0000]  # 640 x 480
    intrin = [283.0345929416784, 0.0, 376.6064871553857, 0.0, 284.3305122630549, 271.0076672594754, 0.0, 0.0, 1]
    trans_base_cam = [-0.409, -0.000, -0.021]
    rot_base_cam = [0.000, 0.000, 1.000, -0.000]
    
    # trans_base_cam = [-1.1102230246251565e-16, 0.020499999999999907, -0.40449]
    # rot_base_cam = [-0.5, 0.4999999999999999, -0.5, -0.5000000000000001]

    # Static tf for hdr camera
    # -0.00451632 -0.09041891 0.04183124 -0.00371707 0.10704935 0.99422573 0.00646622 wide_angle_camera_rear_camera_parent hdr_cam

    # Output settings
    target_shape: Tuple[int, int, int] = (1, 64, 64)
    aux_shape: Tuple[int, int, int] = (1, 64, 64)
    grid_map_resolution: float = 0.1

    def __post_init__(self):
        self.data_dir = os.path.join(self.data_dir_base, self.dataset, self.mode)

data: DataParams = DataParams()
