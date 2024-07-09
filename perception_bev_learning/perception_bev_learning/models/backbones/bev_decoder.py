import torch
import torch.nn as nn

from mmdet.models import BACKBONES
from torchvision.models.resnet import resnet18
from typing import Dict, Any
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop

__all__ = ["BEVDecoder"]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, p_dropout=0.0):
        super().__init__()

        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=p_dropout, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=p_dropout, inplace=False),
        )

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(
            x1, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
        )
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


@BACKBONES.register_module()
class BEVDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p_dropout: float = 0.2,
        *args, **kwargs,
    ):
        super().__init__()

        trunk = resnet18()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, nn.Dropout(p=p_dropout)
        )
        self.layer1 = nn.Sequential(trunk.layer1, nn.Dropout(p=p_dropout))
        self.layer2 = nn.Sequential(trunk.layer2, nn.Dropout(p=p_dropout))
        self.layer3 = nn.Sequential(trunk.layer3, nn.Dropout(p=p_dropout))

        self.up1 = Up(64 + 256, 256, scale_factor=4, p_dropout=p_dropout)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        return [{"micro": x}]
        # return [x]


@BACKBONES.register_module()
class BEVDecoderMulti(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        p_dropout: float = 0.2,
    ):
        super().__init__()

        trunk = resnet18()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, nn.Dropout(p=p_dropout)
        )
        self.layer1 = nn.Sequential(trunk.layer1, nn.Dropout(p=p_dropout))
        self.layer2 = nn.Sequential(trunk.layer2, nn.Dropout(p=p_dropout))
        self.layer3 = nn.Sequential(trunk.layer3, nn.Dropout(p=p_dropout))

        self.up1 = Up(64 + 256, 256, scale_factor=4, p_dropout=p_dropout)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(mid_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        short = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        micro = self.up3(
            nn.functional.interpolate(
                center_crop(short, (128, 128)),
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            )
        )
        return [{"short": short, "micro": micro}]

@BACKBONES.register_module()
class BEVDecoderShort(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p_dropout: float = 0.2,
    ):
        super().__init__()

        trunk = resnet18()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, nn.Dropout(p=p_dropout)
        )
        self.layer1 = nn.Sequential(trunk.layer1, nn.Dropout(p=p_dropout))
        self.layer2 = nn.Sequential(trunk.layer2, nn.Dropout(p=p_dropout))
        self.layer3 = nn.Sequential(trunk.layer3, nn.Dropout(p=p_dropout))

        self.up1 = Up(64 + 256, 256, scale_factor=4, p_dropout=p_dropout)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        short = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )

        return [{"short": short}]

@BACKBONES.register_module()
class BEVDecoderMicro(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        p_dropout: float = 0.2,
    ):
        super().__init__()

        trunk = resnet18()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, nn.Dropout(p=p_dropout)
        )
        self.layer1 = nn.Sequential(trunk.layer1, nn.Dropout(p=p_dropout))
        self.layer2 = nn.Sequential(trunk.layer2, nn.Dropout(p=p_dropout))
        self.layer3 = nn.Sequential(trunk.layer3, nn.Dropout(p=p_dropout))

        self.up1 = Up(64 + 256, 256, scale_factor=4, p_dropout=p_dropout)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(mid_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        micro = self.up3(
            nn.functional.interpolate(
                x,
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            )
        )

        return [{"micro": micro}]

@BACKBONES.register_module()
class BEVDecoderUP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p_dropout: float = 0.2,
        upsample: int = 4,
    ):
        super().__init__()

        trunk = resnet18()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, nn.Dropout(p=p_dropout)
        )
        self.layer1 = nn.Sequential(trunk.layer1, nn.Dropout(p=p_dropout))
        self.layer2 = nn.Sequential(trunk.layer2, nn.Dropout(p=p_dropout))
        self.layer3 = nn.Sequential(trunk.layer3, nn.Dropout(p=p_dropout))

        self.up1 = Up(64 + 256, 256, scale_factor=4, p_dropout=p_dropout)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1, padding=0),
        )
        self.upsample = upsample

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        x = self.up3(
            nn.functional.interpolate(
                x, scale_factor=self.upsample, mode="bilinear", align_corners=True
            )
        )

        return [x]