import shutil
from pathlib import Path
from os.path import join
from dataclasses import asdict
import pickle
import torch

from perception_bev_learning.dataset import get_bev_dataloader
from perception_bev_learning.cfg import ExperimentParams
from perception_bev_learning.utils import get_logger
from perception_bev_learning.lightning import LightningBEV

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler

from pytorch_lightning.strategies import DDPStrategy, BaguaStrategy
from simple_parsing import ArgumentParser
from dataclasses import asdict
import yaml
import os
import random
import time

if __name__ == "__main__":
    """
    Performs full training of the model.
    All parameters are defined in the ExperimentParams.
    Parameters are overwrite using command line arguments:

        python train.py --general.name=test/run

    All results will be stored in the defined - result_dir-folder in cfg/env/xxx.yaml
    xxx is the system variable ENV_WORKSTATION_NAME
    """

    seed_everything(42)

    parser = ArgumentParser()
    parser.confilct_resolver_max_attempts = 100

    parser.add_arguments(ExperimentParams, dest="experiment")
    args = parser.parse_args()
    cfg = args.experiment
    cfg.update()
    shutil.rmtree(cfg.general.model_path, ignore_errors=True)

    Path(cfg.general.model_path).mkdir(parents=True, exist_ok=True)
    with open(join(cfg.general.model_path, "experiment_params.pkl"), "wb") as file:
        pickle.dump(cfg, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(cfg.general.model_path, "experiment_params.yaml"), "w") as file:
        yaml.dump(asdict(cfg), file)

    new_model_path = cfg.general.model_path
    # Setup Logger
    logger = get_logger(cfg)
    asdict(cfg)

    if cfg.logger.name == "neptune":
        logger.run.assign({"ExperimentParams": asdict(cfg)})
        # logger.run["comment"] = cfg.general.comment

    trainer_cfg_dict = asdict(cfg.trainer)
    trainer = Trainer(
        **trainer_cfg_dict, default_root_dir=cfg.general.model_path, logger=logger
    )
    cfg.dataloader_train.shuffle = False
    loader_train, loader_val, loader_test = get_bev_dataloader(
        cfg, return_test_dataloader=True
    )

    ckpt_folder = cfg.trainer_test.ckpt_folder
    ckpt_path = cfg.trainer_test.ckpt_path

    cfg_new = cfg

    # Load a new configuration such that the old configuration for the networks is used
    if ckpt_folder is not None:
        with open(join(ckpt_folder, "experiment_params.pkl"), "rb") as file:
            cfg = pickle.load(file)

            # TODO: Backward competability of old checkpoints - Can be removed in the future!
            cfg.meter.bev_map_size = (500, 500)
            cfg.set_full_logging()
            # Use the new model_path
            cfg.general.model_path = new_model_path
            cfg.meter.fatal_risk = cfg_new.meter.fatal_risk

        checkpoints = [str(s) for s in Path(ckpt_folder).rglob("*.ckpt")]
        checkpoints.sort()
        # checkpoints = [c for c in checkpoints if c.find("val_wheel_risk_cvar_pred_target") != -1]
        ckpt_path = checkpoints[-1]

    # Create model with old_configuration if folder is provided
    module = LightningBEV(cfg)
    for loader, key in zip(
        [loader_train, loader_val, loader_test], ["train", "val", "test"]
    ):
        print(f"Calling Test with dataset-{key} length:   {len(loader.dataset)}")
        module.store_tag = "__" + key
        trainer.test(module, dataloaders=[loader], ckpt_path=ckpt_path)

    print("Done")
