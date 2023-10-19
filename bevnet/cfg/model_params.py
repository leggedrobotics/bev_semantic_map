from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class ModelParams:
    dummy: bool = False

    @dataclass
    class FusionNetParams:
        output_channels: int = 1
        multi_head: bool = False
        anomaly: bool = True
        simple_mlp: bool = False
        apply_sigmoid: List[bool] = field(default_factory=lambda: [True])
        lr: float = 1e-4

    @dataclass
    class LiftSplatShootNetParams:
        @dataclass
        class GridParams:
            xbound: List[float] = field(
                default_factory=lambda: [-3.2, 3.2, 0.1]
            )  # [-51.2, 51.2, 0.2], [-12.8, 12.8, 0.1]
            ybound: List[float] = field(
                default_factory=lambda: [-3.2, 3.2, 0.1]
            )  # [-51.2, 51.2, 0.2], [-12.8, 12.8, 0.1]
            zbound: List[float] = field(default_factory=lambda: [-5.0, 5.0, 10.0])  # [-20.0, 20.0, 40.0]
            dbound: List[float] = field(default_factory=lambda: [1.0, 3.2, 0.05])  # [4.0, 50.0, 0.2]

        @dataclass
        class AugmentationParams:
            H: int = 480  # 396, 128; 540
            W: int = 640  # 640, 128; 720
            fH: int = 480  # 256 (does not work), 640, 128; 512
            fW: int = 640  # 384 (does not work), 480, 128; 640
            resize_lim: List[float] = field(default_factory=lambda: [0.6, 0.7])  # this should be roughly fH/H or fW/W
            bot_pct_lim: List[float] = field(
                default_factory=lambda: [-0.05, 0.05]  # [-0.05, 0.05]
            )  # percentage of scaled image to crop
            rot_lim: List[float] = field(default_factory=lambda: [-5.4, 5.4])  # [-5.4, 5.4]
            rand_flip: bool = False

        grid: GridParams = GridParams()
        augmentation: AugmentationParams = AugmentationParams()
        output_channels: int = 64  # 64
        bevencode: bool = False

    @dataclass
    class PointPillarsParams:
        voxel_size: List[float] = field(default_factory=lambda: [0.1, 0.1, 1.0])  # [0.2, 0.2, 1.0]
        point_cloud_range: List[float] = field(
            default_factory=lambda: [-3.2, -3.2, -2.0, 3.2, 3.2, 2.0]
        )  # [-51.2, -51.2, -10, 51.2, 51.2, 10], [-12.8, -12.8, -10, 12.8, 12.8, 10]
        max_num_points: int = 32  # 32
        max_voxels: Tuple[float] = field(default_factory=lambda: (16000, 40000))  # (16000, 40000)
        output_channels: int = 96  # 96

    fusion_net: FusionNetParams = FusionNetParams()
    lift_splat_shoot_net: LiftSplatShootNetParams = LiftSplatShootNetParams()
    point_pillars: PointPillarsParams = PointPillarsParams()
    # image_backbone: str = "lift_splat_shoot_net"  # If skip, set to "skip"
    image_backbone: str = "skip"  # If skip, set to "skip"
    pointcloud_backbone: str = "point_pillars"  # If skip, set to "skip"
    # pointcloud_backbone: str = "skip"  # If skip, set to "skip"


model: ModelParams = ModelParams()
