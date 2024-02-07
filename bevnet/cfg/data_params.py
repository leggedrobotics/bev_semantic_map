from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch
import os


@dataclass
class DataParams:
    dataset: str = "bevnet2"
    # 2: uetliberg
    # 3: hoengg

    mode: str = "train"
    data_dir_base: str = "/home/rschmid/RosBags"

    # Sensors
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1

    # Image
    img_width: int = 720  # 736 such that it is divisible by 32
    img_height: int = 540  # 544 such that it is divisible by 32

    # Camera parameters
    # intrin = [255.8245, 0.0000, 331.4361, 0.0000, 257.1399, 230.8981, 0.0000, 0.0000, 1.0000]  # 640 x 480
    ## rosrun tf tf_echo wide_angle_camera_rear_camera_parent base
    # trans_base_cam = [-0.409, -0.000, -0.021] # For the front camera
    # rot_base_cam = [0.000, 0.000, 1.000, -0.000]

    intrin = [283.0345929416784, 0.0, 376.6064871553857, 0.0, 284.3305122630549, 271.0076672594754, 0.0, 0.0, 1]    # 720 x 540
    trans_base_cam = [-0.000, 0.020, -0.404]
    rot_base_cam = [0.500, 0.500, -0.500, 0.500]

    # Output settings
    target_shape: Tuple[int, int, int] = (1, 64, 64)
    aux_shape: Tuple[int, int, int] = (1, 64, 64)
    grid_map_resolution: float = 0.1

    def __post_init__(self):
        self.data_dir = os.path.join(self.data_dir_base, self.dataset, self.mode)

data: DataParams = DataParams()
