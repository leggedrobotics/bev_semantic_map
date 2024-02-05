from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any


@dataclass
class ModelParams:
    dummy: bool = False

    @dataclass
    class FusionNetParams:
        pass

    @dataclass
    class CNNParams:
        output_channels: int = 2

    @dataclass
    class LiftSplatShootNetParams:
        @dataclass
        class GridParams:
            xbound: List[float] = field(
                default_factory=lambda: [-3.2, 3.2, 0.1]
            )  # [-51.2, 51.2, 0.2]
            ybound: List[float] = field(
                default_factory=lambda: [-3.2, 3.2, 0.1]
            )  # [-51.2, 51.2, 0.2]
            zbound: List[float] = field(default_factory=lambda: [-5.0, 5.0, 10.0])  # [-20.0, 20.0, 40.0]
            dbound: List[float] = field(default_factory=lambda: [1.0, 3.2, 0.05])  # [4.0, 50.0, 0.2]

        @dataclass
        class AugmentationParams:
            H: int = 540  # 396, 128; 540
            W: int = 720  # 640, 128; 720
            fH: int = 512  # 256 (does not work), 640, 128; 512 Images need to be divisible by 32
            fW: int = 704  # 384 (does not work), 480, 128; 640 Images need to be divisible by 32
            resize_lim: List[float] = field(default_factory=lambda: [0.6, 0.7])  # this should be roughly fH/H or fW/W
            bot_pct_lim: List[float] = field(
                default_factory=lambda: [-0.05, 0.05]  # [-0.05, 0.05]
            )  # percentage of scaled image to crop
            rot_lim: List[float] = field(default_factory=lambda: [-5.4, 5.4])  # [-5.4, 5.4]
            rand_flip: bool = False

        grid: GridParams = GridParams()
        augmentation: AugmentationParams = AugmentationParams()
        # TODO: add augmentations
        output_channels: int = 64  # 64
        bevencode: bool = False

    @dataclass
    class PointPillarsParams:
        voxel_size: List[float] = field(default_factory=lambda: [0.1, 0.1, 4.0])  # [0.2, 0.2, 1.0]
        point_cloud_range: List[float] = field(
            default_factory=lambda: [-3.2, -3.2, -2.0, 3.2, 3.2, 2.0]
        )  # [-51.2, -51.2, -10, 51.2, 51.2, 10]
        max_num_points: int = 32  # 32
        max_voxels: Tuple[float] = field(default_factory=lambda: (16000, 40000))  # (16000, 40000)
        output_channels: int = 96  # 96

    lift_splat_shoot_net: LiftSplatShootNetParams = LiftSplatShootNetParams()
    point_pillars: PointPillarsParams = PointPillarsParams()
    fusion_net = FusionNetParams()

    image_backbone: str = "lift_splat_shoot_net"  # "lift_splat_shoot_net" or "skip
    # image_backbone: str = "skip"  # "lift_splat_shoot_net" or "skip
    pointcloud_backbone: str = "point_pillars"   # "point_pillars" or "skip"
    # pointcloud_backbone: str = "skip"  # "point_pillars" or "skip"
    fusion_backbone: str = "CNN"    # "CNN" or "skip"

    def __post_init__(self):
        if self.fusion_backbone != "skip":
            self.fusion_net = self.CNNParams()


# Do not change below here
model: ModelParams = ModelParams()
