from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class DataParams:
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1
    nr_data: int = 100

    img_width: int = 128    # 640
    img_height: int = 128   # 480

    img_path: str = "/home/rschmid/RosBags/bevnet/image"
    pcd_path: str = "/home/rschmid/RosBags/bevnet/pcd"
    target_path: str = "/home/rschmid/RosBags/bevnet/target"

    trans_pc_cam = [0.025654243139211275, 0.0406744001863073, -0.004784660744370228]
    rot_pc_cam = [0.0010163463332151373, 0.1270025471498098, 0.9918701104718138, 0.007937506549489694]

    nr_points: int = 5000

    target_shape: Tuple[int, int, int] = (1, 128, 128)
    aux_shape: Tuple[int, int, int] = (1, 128, 128)

    gird_map_resolution: float = 0.1


data: DataParams = DataParams()
