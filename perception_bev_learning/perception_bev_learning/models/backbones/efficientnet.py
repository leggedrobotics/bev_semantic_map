from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from typing import List
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class EfficientNetB(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        out_indices: List[int] = [2, 3, 4],
        drop_rate: float = 0,
        freeze: bool = False,
    ):
        super().__init__()
        self.trunk = EfficientNet.from_pretrained(model_name)
        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self.freeze = freeze
        # EfficientNetB0 has 15 blocks (Block0 - Block15)
        # The reductions happen after blocks -> [0, 2, 7, 12, 15]
        # out_indices = [2, 3, 4] imply -> taking inputs of indices 2,3,4 from above list

    def init_weights(self):
        # It is always loaded with pretrained weights
        # TODO -> Freezing stages
        if self.freeze:
            for name, param in self.trunk.named_parameters():
                param.requires_grad = False
        pass

    def forward(self, x):
        features = []
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.drop_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.trunk._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                # Concatenate the prev feature before the reduction
                features.append(prev_x)
            prev_x = x

        # Concatenate the last block pred
        features.append(x)

        return [features[i] for i in self.out_indices]
