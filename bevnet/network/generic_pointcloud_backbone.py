from abc import ABC, abstractmethod
from typing import Any
from typing import Tuple, Dict, List, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
from pytictac import ClassTimer


class GenericPointcloudBackbone(nn.Module):
    def __init__(self):
        super(GenericPointcloudBackbone, self).__init__()
        self._cct = ClassTimer(objects=[self], names=["Backbone"], enabled=False)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, scan: torch.Tensor):
        """Forward pass of the network

        Args:
            x (torch.Tensor, shape=(NR_POINTS, 3+C), dtype=np.float16/32):
                Input tensor representing point cloud data, where:
                - NR_POINTS: number of points
                - 3: coordinates of each point (x, y, z)
                - C: additional features for each point
            batch (torch.Tensor, shape=(NR_BATCHES), dtype=np.int32):
                Index tensor representing the batch to which each point belongs, where:
                - NR_BATCHES: number of batches
            scan (torch.Tensor, shape (NR_SCANS), dtype=np.int32):
                Index tensor representing the scan to which each point belongs, where:
                - NR_SCANS: number of scans

        Returns:
            torch.Tensor: Output feature emmbedding
        """
        x = self.preprocessing(x, batch, scan)
        x = self.embed(x)
        x = self.postprocessing(x)
        return x

    @abstractmethod
    def preprocessing(self, x: torch.Tensor, batch: torch.Tensor, scan: torch.Tensor):
        pass

    @abstractmethod
    def embed(self, x: Any):
        pass

    @abstractmethod
    def postprocessing(self, x: Any):
        pass
