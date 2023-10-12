from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class RunParams:
    wandb_logging: bool = False
    training_batch_size: int = 4
    epochs: int = 5


data: RunParams = RunParams()
