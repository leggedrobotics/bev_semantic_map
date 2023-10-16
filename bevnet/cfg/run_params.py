from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class RunParams:
    wandb_logging: bool = True
    training_batch_size: int = 8
    epochs: int = 2


data: RunParams = RunParams()
