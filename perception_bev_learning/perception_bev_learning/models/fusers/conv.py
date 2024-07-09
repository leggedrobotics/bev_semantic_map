from typing import List

import torch
from torch import nn

from perception_bev_learning.models.builder import FUSERS
from perception_bev_learning.models.vtransforms import ASPP

__all__ = ["ConvFuser", "ConvFuserUpsample", "ConvFuserASPP"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@FUSERS.register_module()
class ConvFuserUpsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, upsample: int = 1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.upsample = upsample

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return nn.functional.interpolate(
            super().forward(torch.cat(inputs, dim=1)),
            scale_factor=self.upsample,
            mode="bilinear",
        )


@FUSERS.register_module()
class ConvFuserASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: int = 1,
        mid_channels: int = 256,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.upsample = upsample
        self.mid_channels = mid_channels
        self.convfuser = nn.Sequential(
            nn.Conv2d(sum(in_channels), mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.aspp = ASPP(inplanes=mid_channels, mid_channels=out_channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = nn.functional.interpolate(
            self.convfuser(torch.cat(inputs, dim=1)),
            scale_factor=self.upsample,
            mode="bilinear",
        )
        x = self.aspp(x)

        return x
