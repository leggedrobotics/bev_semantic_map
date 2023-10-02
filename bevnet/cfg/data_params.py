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

    img_path: str = "/home/rschmid/RosBags/bevnet/image"    # Camera frame
    pcd_path: str = "/home/rschmid/RosBags/bevnet/pcd"  # Base frame
    target_path: str = "/home/rschmid/RosBags/bevnet/mask"  # Base frame

    # trans_pc_cam = [0.025654243139211275, 0.0406744001863073, -0.004784660744370228]
    # rot_pc_cam = [0.0010163463332151373, 0.1270025471498098, 0.9918701104718138, 0.007937506549489694]

    intrin = [575.6050407221768, 0.0, 745.7312198525915, 0.0, 578.564849365178, 519.5207040671075, 0.0, 0.0, 1.0]
    trans_base_cam = [0.40449, 0.0, 0.0205]
    rot_base_cam = [0.5, -0.4999999999999999, 0.5, -0.5000000000000001]

    nr_points: int = 5000

    target_shape: Tuple[int, int, int] = (1, 128, 128)
    aux_shape: Tuple[int, int, int] = (1, 128, 128)

    gird_map_resolution: float = 0.1


data: DataParams = DataParams()
