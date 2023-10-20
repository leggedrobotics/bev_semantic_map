from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class RunParams:
    training_batch_size: int = 4
    epochs: int = 1
    lr: float = 1e-4
    log_name: str = "bevnet5"


data: RunParams = RunParams()
