from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class DinoV2(nn.Module):
    def __init__(
        self,
        unfreeze_layers: List = [],
        out_indices: List = [12],
        return_cls_token: bool = False,
        size: str = "small",
        output_downsample: float = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        if size == "small":
            size = "dinov2_vits14"
        elif size == "base":
            size = "dinov2_vitb14"
        self.model = torch.hub.load("facebookresearch/dinov2", size)

        self.patch_size = 14
        self.return_cls_token = return_cls_token
        self.layer_output = out_indices
        self.unfreeze_layers = unfreeze_layers
        self.output_downsample = output_downsample

    def init_weights(self):
        self.layer_output = [i - 1 for i in self.layer_output]
        self.unfreeze_layers = [i - 1 for i in self.unfreeze_layers]

        for name, param in self.model.named_parameters():
            # print(name)
            layer_vals = [
                name.startswith("blocks.{}".format(i - 1)) for i in self.unfreeze_layers
            ]
            if any(layer_vals):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        # convert the input to be divisible by patch size
        B, C, H, W = x.shape

        x = F.interpolate(
            x,
            size=(
                H // self.patch_size * self.patch_size,
                W // self.patch_size * self.patch_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        outputs = self.model.get_intermediate_layers(
            x,
            self.layer_output,
            reshape=True,
            return_class_token=self.return_cls_token,
        )

        # Upsample the outputs
        out_shape = [H // self.output_downsample, W // self.output_downsample]
        up_out = []
        for i, out in enumerate(outputs):
            up_out.append(F.interpolate(out, size=(out_shape[0], out_shape[1])))

        return up_out
