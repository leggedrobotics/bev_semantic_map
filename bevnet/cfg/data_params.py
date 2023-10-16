from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class DataParams:
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1
    nr_data: int = 100

    img_width: int = 640    # 640, 128; 720
    img_height: int = 480   # 480, 128; 540

    data_dir: str = "/home/rschmid/RosBags/bevnet"

    # trans_pc_cam = [0.025654243139211275, 0.0406744001863073, -0.004784660744370228]
    # rot_pc_cam = [0.0010163463332151373, 0.1270025471498098, 0.9918701104718138, 0.007937506549489694]

    # intrin = [287.8025, 0.0000, 372.8656, 0.0000, 289.2824, 259.7603, 0.0000, 0.0000, 1.0000]   # 720 x 540
    intrin = [255.8245, 0.0000, 331.4361, 0.0000, 257.1399, 230.8981, 0.0000, 0.0000, 1.0000]   # 640 x 480
    trans_base_cam = [0.40449, 0.0, 0.0205]
    rot_base_cam = [0.5, -0.4999999999999999, 0.5, -0.5000000000000001]

    nr_points: int = 5000

    target_shape: Tuple[int, int, int] = (1, 64, 64)
    aux_shape: Tuple[int, int, int] = (1, 64, 64)

    grid_map_resolution: float = 0.1


data: DataParams = DataParams()
