from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as pl
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from perception_bev_learning.utils.lightning_utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from torch.utils.benchmark import Timer
import time
import torch
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.general.get("seed"):
        pl.seed_everything(cfg.general.seed, workers=True)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Save the cfg / param file ss

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.general.train:
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.general.get("ckpt_path")
        )
    # if cfg.general.train:
    #     log.info("Starting Dataloading!")
    #     datamodule.setup()
    #     # for idx, batch in enumerate(datamodule.train_dataloader()):
    #     #     # Do nothing, just iterate over the batches to benchmark data loading time
    #     #     pass
    #     #     print(idx)
    #     # dataloader_time = timer_dataloader.blocked_autorange(min_run_time=1)

    #     # print(f"Data Loader Time: {dataloader_time}")
    #     # Set the number of iterations you want to benchmark (e.g., 100)
    #     num_iterations = 20

    #     # Record the start time
    #     start_time = time.time()
    #     train_dataloader = datamodule.train_dataloader()
    #     # Iterate over the DataLoader for a fixed number of iterations
    #     for iteration in range(num_iterations):
    #         # Get a batch from the DataLoader
    #         batch = next(iter(train_dataloader))
    #         print(iteration)
    #         # Add any processing you want to do with the batch (optional)
    #         # ...

    #     # Calculate the elapsed time
    #     elapsed_time = time.time() - start_time

    #     # Calculate and print the average time per iteration
    #     avg_time_per_iteration = elapsed_time / num_iterations
    #     print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")



    train_metrics = trainer.callback_metrics

    if cfg.general.test:
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
