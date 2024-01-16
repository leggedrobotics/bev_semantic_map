from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    training_batch_size: int = 8
    epochs: int = 1000
    lr: float = 1e-2
    weight_decay: float = 1e-5
    log_name: str = "bevnet_discriminative"


data: RunParams = RunParams()
