from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class ModelParams:
    dummy: bool = False

    @dataclass
    class FusionNetParams:
        output_channels: int = 1
        multi_head: bool = True
        apply_sigmoid: List[bool] = field(default_factory=lambda: [True])
    
    @dataclass
    class LiftSplatShootNetParams:
        @dataclass
        class GridParams:
            xbound: List[float] = field(default_factory=lambda: [-51.2, 51.2, 0.2])
            ybound: List[float] = field(default_factory=lambda: [-51.2, 51.2, 0.2])
            zbound: List[float] = field(default_factory=lambda: [-20.0, 20.0, 40.0])
            dbound: List[float] = field(default_factory=lambda: [4.0, 50.0, 0.2])
        
        @dataclass
        class AugmentationParams:
            H: int = 396
            W: int = 640
            fH: int = 256
            fW: int = 384
            resize_lim: List[float] = field(default_factory=lambda: [0.6, 0.7])  # this should be roughly fH/H or fW/W
            bot_pct_lim: List[float] = field(
                default_factory=lambda: [-0.05, 0.05]
            )  # percentage of scaled image to crop
            rot_lim: List[float] = field(default_factory=lambda: [-5.4, 5.4])
            rand_flip: bool = False
            
        grid: GridParams = GridParams()
        augmentation: AugmentationParams = AugmentationParams()
        output_channels: int = 64
        bevencode: bool = False

    @dataclass
    class PointPillarsParams:
        voxel_size: List[float] = field(default_factory=lambda: [0.2, 0.2, 1.0])
        point_cloud_range: List[float] = field(default_factory=lambda: [-51.2, -51.2, -10, 51.2, 51.2, 10])
        max_num_points: int = 32
        max_voxels: Tuple[float] = field(default_factory=lambda: (16000, 40000))
        output_channels: int = 96

    fusion_net: FusionNetParams = FusionNetParams()
    lift_splat_shoot_net: LiftSplatShootNetParams = LiftSplatShootNetParams()
    point_pillars: PointPillarsParams = PointPillarsParams()
    image_backbone: str = "lift_splat_shoot_net" 
    pointcloud_backbone: str = "point_pillars"
    
model: ModelParams = ModelParams()
