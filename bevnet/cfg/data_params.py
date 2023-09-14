from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class DataParams:
    nr_cameras: int = 1
    nr_lidar_points_time: int = 1
    nr_data: int = 100

    img_width: int = 640
    img_height: int = 480

    nr_points: int = 5000

    target_shape: Tuple[int, int, int] = (1, 256, 256)
    aux_shape: Tuple[int, int, int] = (1, 256, 256)

    gird_map_resolution: float = 0.1


data: DataParams = DataParams()
