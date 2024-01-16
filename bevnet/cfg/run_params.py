from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    nr_data: int = 160 # -1 for all
    training_batch_size: int = 8
    epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 1e-5
    log_name: str = "bevnet_geometry"


data: RunParams = RunParams()
