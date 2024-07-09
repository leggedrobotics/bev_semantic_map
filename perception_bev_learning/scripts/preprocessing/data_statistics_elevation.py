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

EPS = 0.001

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

    nr_bins = 256
    value_bins = torch.zeros((nr_bins,))
    for j, batch in enumerate(loader_train):
        (imgs, rots, trans, intrins, post_rots, post_trans, target, *_) = batch

        target = target[:, 1]
        bin_low, bin_high = -1, 1
        m = torch.isnan(target)
        with torch.no_grad():
            target_bins = target.clip(bin_low, bin_high) - bin_low
            target_bins /= bin_high - bin_low + EPS

            target_bins *= nr_bins
            target_bins = target_bins.type(torch.long)

        vals, count = torch.unique(target_bins[~m], return_counts=True)
        value_bins[vals] += count

        leg = len(loader_train)
        print(f"{j}/{leg}")

    res = value_bins / value_bins.sum()
    torch.save(res, join(BEV_ROOT_DIR, "assets", "elevation_clip_-1_1_scale_0.05_prob_cluster.pt"))

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

    import matplotlib

    matplotlib.use("agg")
    from torch import nn
    import matplotlib.pyplot as plt

    val = res
    fig, ax = plt.subplots()

    labels = [f"bin_{x}" for x in range(val.shape[0])]
    counts = val.cpu().numpy()
    ax.bar(labels, counts, label=labels)

    ax.set_ylabel("Prob")
    ax.set_title("Distribution of elevation_values in dataset")
    ax.legend(title="Probability")

    from perception_bev_learning.visu import get_img_from_fig

    img = get_img_from_fig(fig)
    import imageio

    imageio.imwrite("/tmp/img.png", img)

    plt.show()

    print("Done")
