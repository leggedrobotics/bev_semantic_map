from dataclasses import dataclass, field, asdict


@dataclass
class RunParams:
    nr_data: int = -1   # -1 for all
    training_batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_name: str = "img_geom_dense"


data: RunParams = RunParams()
