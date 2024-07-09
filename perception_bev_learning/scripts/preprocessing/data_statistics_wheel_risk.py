import torch
from time import time
import numpy as np
import os
from dataclasses import asdict
from perception_bev_learning.dataset import get_bev_dataloader
from perception_bev_learning.cfg import ExperimentParams
from perception_bev_learning.utils import denormalize_img
from pytorch_lightning import seed_everything
from perception_bev_learning import BEV_ROOT_DIR
from os.path import join


if __name__ == "__main__":
    """
    Script used to compute the statistics across the training dataset.
    Mainly used to create a balanced loss function!
    """

    seed_everything(42)

    cfg = ExperimentParams()
    cfg.update()
    cfg.dataloader_train.batch_size = 1
    loader_train, loader_val, loader_test = get_bev_dataloader(cfg)

    # counter = {d.name: {"nan":[], "not_nan": [], "mean" : [], "max":[], "min":[]} for d in cfg.dataset_train.target_layers}
    # ma = 0

    value_bins = torch.zeros((100,))
    for j, batch in enumerate(loader_train):
        (imgs, rots, trans, intrins, post_rots, post_trans, target, *_) = batch
        m = ~torch.isnan(target)
        bins = (target[m] * 99).type(torch.long)
        vals, count = torch.unique(bins, return_counts=True)
        value_bins[vals] += count
        leg = len(loader_train)
        print(f"{j}/{leg}")
    res = value_bins / value_bins.sum()
    torch.save(res, join(BEV_ROOT_DIR, "assets", "wheel_risk_cvar_0_1_prob_cluster.pt"))

    #     non_nan = ~torch.isnan(target[:,1,:,:])
    #     ma = max(ma, target[:,1,:,:][non_nan].max())
    #     print(ma)
    #     for n, t in zip(counter.keys(), target[0]):
    #         m = (~torch.isnan( t ))
    #         counter[n]["nan"].append( torch.isnan( t ).sum())
    #         counter[n]["not_nan"].append((~torch.isnan( t )).sum())
    #         counter[n]["mean"].append( torch.mean(t[m]))
    #         counter[n]["max"].append( torch.max( t[m]))
    #         counter[n]["min"].append( torch.min( t[m]))
    # print("Summary:")
    # for k,v in counter.items():
    #     print(k)
    #     for k1,v1 in v.items():
    #         print(f"    {k1}: {torch.stack(v1).type(torch.float32).mean()}")
    print("Done")
