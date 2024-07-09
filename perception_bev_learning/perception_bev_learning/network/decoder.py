from torchvision.models.resnet import resnet18
import torch
from torch import nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(
            x1, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
        )
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class DecoderNet(nn.Module):
    def __init__(self, cfg_decoder):
        super(DecoderNet, self).__init__()

        self._cfg_decoder = cfg_decoder

        if self._cfg_decoder.small_model:
            Decoder = SmallDecoder
        else:
            Decoder = BevEncode

        if self._cfg_decoder.multi_head:
            self.wheel_risk = Decoder(
                cfg_decoder.input_channels,
                self._cfg_decoder.channels_wheel_risk,
                p_dropout=cfg_decoder.p_dropout,
            )
            self.elevation = Decoder(
                cfg_decoder.input_channels,
                self._cfg_decoder.channels_elevation,
                p_dropout=cfg_decoder.p_dropout,
            )

        else:
            self.shared = Decoder(
                cfg_decoder.input_channels,
                self._cfg_decoder.channels_wheel_risk
                + self._cfg_decoder.channels_elevation,
                p_dropout=cfg_decoder.p_dropout,
            )

    def forward(self, x):
        if self._cfg_decoder.multi_head:
            return {
                "wheel_risk": self.wheel_risk(x).contiguous(),
                "elevation": self.elevation(x).contiguous(),
            }
        else:
            out = self.shared(x)
            return {
                "wheel_risk": out[
                    :, : self._cfg_decoder.channels_wheel_risk
                ].contiguous(),
                "elevation": out[
                    :, self._cfg_decoder.channels_wheel_risk :
                ].contiguous(),
            }


class SmallDecoder(nn.Module):
    def __init__(self, inC, outC, p_dropout=0.0):
        super(SmallDecoder, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layer(x)


class BevEncode(nn.Module):
    def __init__(self, inC, outC, p_dropout=0.0):
        super(BevEncode, self).__init__()
        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        # Currently not used
        self.apply_dropout = p_dropout > 0.0
        if self.apply_dropout:
            self.drop = torch.nn.Dropout(p=0.2, inplace=False)
            self.conv1 = nn.Sequential(self.conv1, self.drop)
            self.layer1 = nn.Sequential(self.layer1, self.drop)
            self.layer2 = nn.Sequential(self.layer2, self.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(
            nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        return x
