from pathlib import Path
from dataclasses import asdict
import torch

from perception_bev_learning.dataset import get_bev_dataloader
from perception_bev_learning.cfg import ExperimentParams
from simple_parsing import ArgumentParser
from dataclasses import asdict
import yaml
import os

if __name__ == "__main__":
    """
    Performs full training of the model.
    All parameters are defined in the ExperimentParams.
    Parameters are overwrite using command line arguments:

        python train.py --general.name=test/run

    All results will be stored in the defined - result_dir-folder in cfg/env/xxx.yaml
    xxx is the system variable ENV_WORKSTATION_NAME
    """

    parser = ArgumentParser()
    parser.confilct_resolver_max_attempts = 100

    parser.add_arguments(ExperimentParams, dest="experiment")
    args = parser.parse_args()
    cfg = args.experiment
    cfg.update()

    loader_train, loader_val = get_bev_dataloader(cfg, return_test_dataloader=False)

    import lovely_tensors as lt

    lt.monkey_patch()

    for j, batch in enumerate(loader_train):
        imgs, rots, trans, intrins, post_rots, post_trans, target, aux, *_, pcd_new = batch
        print(str(j) + "\n\n")
        for t in [imgs, rots, trans, intrins, post_rots, post_trans, target, aux]:
            print(t)
