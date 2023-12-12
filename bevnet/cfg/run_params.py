from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    training_batch_size: int = 1
    epochs: int = 100
    lr: float = 1e-4
    log_name: str = "bevnet_discriminative"


data: RunParams = RunParams()
