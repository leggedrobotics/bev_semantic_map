from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    training_batch_size: int = 2
    epochs: int = 50
    lr: float = 1e-4
    log_name: str = "bevnet_discriminative"


data: RunParams = RunParams()
