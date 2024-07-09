from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from perception_bev_learning.utils import flatten_dict
import inspect
import os
from dataclasses import asdict
from perception_bev_learning import BEV_ROOT_DIR
from pathlib import Path
from typing import List
import neptune

__all__ = [
    "get_neptune_logger",
    "get_wandb_logger",
    "get_tensorboard_logger",
    "get_neptune_run",
]

PROXIES = {"http": "http://proxy.ethz.ch:3128", "https": "http://proxy.ethz.ch:3128"}


def get_neptune_run(neptune_project_name: str, tags: List[str]) -> any:
    """Get neptune run

    Args:
        neptune_project_name (str): Neptune project name
        tags (list of str): Tags to identify the project
    """
    raise ValueError("not implemented for latest API")

    proxies = None
    if os.environ["ENV_WORKSTATION_NAME"] == "euler":
        proxies = PROXIES

    run = neptune.init(
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        project=neptune_project_name,
        tags=[os.environ["ENV_WORKSTATION_NAME"]] + tags,
        proxies=proxies,
    )
    return run


def get_neptune_logger(cfg: any) -> NeptuneLogger:
    """Returns NeptuneLogger

    Args:
        cfg (any): Experiment Paramters
    Returns:
        (logger): Logger
    """
    project_name = (
        cfg.logger.neptune_project_name
    )  # Neptune AI project_name "username/project"

    params = flatten_dict(asdict(cfg))

    name_full = cfg.general.name
    name_short = "__".join(name_full.split("/")[-2:])

    proxies = None
    if os.environ["ENV_WORKSTATION_NAME"] == "euler":
        proxies = PROXIES
    pa = Path(BEV_ROOT_DIR)
    py = [str(s) for s in pa.rglob("*.py")]
    yaml = [str(s) for s in pa.rglob("*.yaml")]
    sh = [str(s) for s in pa.rglob("*.sh")]

    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=project_name,
        name=name_short,
        tags=[
            os.environ["ENV_WORKSTATION_NAME"],
            name_full.split("/")[-2],
            name_full.split("/")[-1],
        ],
        # tags=[os.environ["ENV_WORKSTATION_NAME"], name_full.split("/")[-2], name_full.split("/")[-1]],
        proxies=proxies,
        source_files=py + yaml + sh,
    )

    logger.run["comment"] = cfg.general.comment
    logger.run["parameters"] = params

    return logger


def get_wandb_logger(cfg: any) -> WandbLogger:
    """Returns NeptuneLogger

    Args:
        cfg (any): Experiment Paramters

    Returns:
        (logger): Logger
    """
    project_name = cfg.logger.wandb_project_name  # project_name (str): W&B project_name
    save_dir = os.path.join(
        cfg.general.model_path
    )  # save_dir (str): File path to save directory
    params = flatten_dict(asdict(cfg))
    name_full = cfg.general.name
    name_short = "__".join(name_full.split("/")[-2:])
    return WandbLogger(
        name=name_short,
        project=project_name,
        entity=cfg.logger.wandb_entity,
        save_dir=save_dir,
        offline=False,
    )


def get_tensorboard_logger(cfg: any) -> TensorBoardLogger:
    """Returns TensorboardLoggers

    Args:
        cfg (any): Experiment Paramters

    Returns:
        (logger): Logger
    """
    params = flatten_dict(asdict(cfg))
    return TensorBoardLogger(
        save_dir=cfg.general.model_path, name="tensorboard", default_hp_metric=params
    )


def get_skip_logger(cfg: any) -> None:
    """Returns None

    Args:
        cfg (any): Experiment Paramters

    Returns:
        (logger): Logger
    """
    return None


def get_logger(cfg: any) -> any:
    name = cfg.logger.name
    register = {k: v for k, v in globals().items() if inspect.isfunction(v)}
    return register[f"get_{name}_logger"](cfg)
