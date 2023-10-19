from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class RunParams:
    training_batch_size: int = 4
    epochs: int = 1


data: RunParams = RunParams()
