# %%

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
import matplotlib
from torch import nn
import matplotlib.pyplot as plt
from perception_bev_learning.visu import get_img_from_fig
from perception_bev_learning.visu import paper_colors_rgba_f
import imageio
import seaborn as sns

# %%

res = torch.load(join(BEV_ROOT_DIR, "assets", "elevation_clip_-1_1_scale_0.05_prob_cluster.pt"))
matplotlib.use("qtagg")
val = res


fig, ax = plt.subplots()
labels = [f"{round(x,1)}" for x in np.arange(-25, 25 + 50 / 255, 50 / 255)]
counts = val.cpu().numpy()
ax.bar(labels, counts, align="edge", width=1.0, color=paper_colors_rgba_f["cyan"])
ax.set_ylabel("Prob")
ax.set_title("wheel_risk_cvar")
ax.legend(title="Probability")
ax.set_xticklabels
ticks = ax.get_xticks()
ax.set_xticks(ticks[::51])
ticks = ax.get_yticks()
ax.set_yticks(ticks[::2])


ax2 = ax.twinx()

x = np.arange(-25, 25 + 50 / 255, 50 / 255)
HISTOGRAM_ELEVATION = torch.load(join(BEV_ROOT_DIR, "assets", "elevation_clip_-1_1_scale_0.05_prob_cluster.pt"))
HISTOGRAM_ELEVATION_EQ = torch.ones_like(HISTOGRAM_ELEVATION) / HISTOGRAM_ELEVATION.shape[0]

HISTOGRAM_RISK = torch.load(join(BEV_ROOT_DIR, "assets", f"wheel_risk_cvar_0_1_prob_cluster.pt"))
HISTOGRAM_RISK_EQ = torch.ones_like(HISTOGRAM_RISK) / HISTOGRAM_RISK.shape[0]
bin_scaling = HISTOGRAM_ELEVATION + HISTOGRAM_ELEVATION_EQ * 0.05
bin_scaling /= bin_scaling.sum()
bin_scaling = bin_scaling
WEIGHT_ELEVATION = 1 - bin_scaling
bin_scaling = HISTOGRAM_RISK + HISTOGRAM_RISK_EQ * 0.2
bin_scaling /= bin_scaling.sum()
bin_scaling = bin_scaling
WEIGHT_RISK = 1 - bin_scaling
y = WEIGHT_ELEVATION
ax2.plot(x, y, color="r", marker="o", label="Line")

img = get_img_from_fig(fig)
plt.show()
# imageio.imwrite("/tmp/img.png", img)

# plt.show()
print(x)
# print("Done")

# %%
