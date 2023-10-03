from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any
import torch


@dataclass
class RunParams:
    wandb_logging: bool = False
    training_batch_size: int = 1    # TODO: make running with batch size > 1 work


data: RunParams = RunParams()
