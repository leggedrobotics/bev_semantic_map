import torch
from torch import nn
from perception_bev_learning.models.builder import HEADS


@HEADS.register_module()
class BEVSimpleHead(nn.Module):
    def __init__(
        self, in_channels, channels_wheel_risk, channels_elevation, scale_factor
    ) -> None:
        super().__init__()

        inC = in_channels
        outC = channels_wheel_risk + channels_elevation
        self.channels_wheel_risk = channels_wheel_risk
        self.channels_elevation = channels_elevation
        self.scale_factor = scale_factor
        self.up = nn.Sequential(
            nn.Conv2d(inC, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        out = self.up(
            nn.functional.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )
        )

        # return out
        return {
            "wheel_risk": out[:, : self.channels_wheel_risk].contiguous(),
            "elevation": out[:, self.channels_wheel_risk :].contiguous(),
        }


@HEADS.register_module()
class BEVMultiHead(nn.Module):
    def __init__(self, in_channels, target_layers) -> None:
        super().__init__()

        self.out_channel_dict = {}
        self.out_layer_name = {}
        self.up = nn.ModuleDict()

        for gridmap_key in target_layers.keys():
            self.out_channel_dict[gridmap_key] = 0
            self.out_layer_name[gridmap_key] = {}
            for lname, layers in target_layers[gridmap_key].items():
                self.out_layer_name[gridmap_key][lname] = layers["channels"]
                self.out_channel_dict[gridmap_key] += layers["channels"]

            self.up[gridmap_key] = nn.Sequential(
                nn.Conv2d(
                    in_channels[gridmap_key], 64, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    64, self.out_channel_dict[gridmap_key], kernel_size=1, padding=0
                ),
            )

    def forward(self, x):
        output = {}

        for gridmap_key in self.out_layer_name.keys():
            output[gridmap_key] = {}
            out = self.up[gridmap_key](x[gridmap_key])
            prev_channel = 0
            for layer, channels in self.out_layer_name[gridmap_key].items():
                output[gridmap_key][layer] = out[
                    :, prev_channel : prev_channel + channels
                ].contiguous()
                prev_channel = prev_channel + channels

        return output
