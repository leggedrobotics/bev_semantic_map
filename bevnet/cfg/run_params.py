from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    nr_data: int = 10 # -1 for all
    training_batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-5
    log_name: str = "bevnet_discriminative"


data: RunParams = RunParams()
