import os
import torch
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any


@dataclass
class DataParams:
    dataset: str = "bevnet_perugia"
    # 2: uetliberg
    # 3: hoengg

    mode: str = "train"
    data_dir_base: str = "/home/rschmid/RosBags"

    # Sensors
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1

    # Wide angle camera
    # img_width: int = 640  # 736 such that it is divisible by 32
    # img_height: int = 480  # 544 such that it is divisible by 32

    # HDR camera
    img_width: int = 720  # 736 such that it is divisible by 32
    img_height: int = 540  # 544 such that it is divisible by 32

    # Camera parameters
    # Wide angle rear camera parameters
    # intrin = [255.8245, 0.0000, 331.4361, 0.0000, 257.1399, 230.8981, 0.0000, 0.0000, 1.0000]  # 640 x 480
    ## rosrun tf tf_echo wide_angle_camera_rear_camera_parent base
    # trans_base_cam = [-0.409, -0.000, -0.021] # For the front camera
    # rot_base_cam = [0.000, 0.000, 1.000, -0.000]

    # Alphasense camera parameters for Perugia
    # rosrun tf tf_echo cam4_sensor_frame_helper base
    intrin = [186.00912763958874, 0.0, 331.7788889547211, 0.0, 185.95679342313744, 257.5388656297653, 0.0, 0.0, 1.0]
    trans_base_cam = [-0.007, 0.229, -0.408]
    rot_base_cam = [-0.492, -0.506, 0.508, -0.494]

    # Wide angle front camera parameters for Hoengg old
    ## rosrun tf tf_echo wide_angle_camera_front_camera_parent base
    # intrin = [287.80252036, 0.0, 372.86560993, 0.0, 289.28242468, 259.76035203, 0.0, 0.0, 1.0]
    # trans_base_cam = [-0.000, 0.020, -0.404]
    # rot_base_cam = [0.500, -0.500, 0.500, 0.500]

    # HDR camera parameters
    # intrin = [283.0345929416784, 0.0, 376.6064871553857, 0.0, 284.3305122630549, 271.0076672594754, 0.0, 0.0, 1]    # 720 x 540 (HDR)
    # trans_base_cam = [-0.000, 0.020, -0.404]
    # rot_base_cam = [0.500, 0.500, -0.500, 0.500]

    # Output settings
    target_shape: Tuple[int, int, int] = (1, 64, 64)
    aux_shape: Tuple[int, int, int] = (1, 64, 64)
    grid_map_resolution: float = 0.1

    def __post_init__(self):
        self.data_dir = os.path.join(self.data_dir_base, self.dataset, self.mode)

data: DataParams = DataParams()
