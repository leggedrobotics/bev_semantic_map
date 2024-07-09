import numpy as np
import matplotlib.colors as mcolors

safe_risk = 0.0
watchful_risk = 0.13
critical_risk = 0.35
fatal_risk = 0.35

safe_color = np.array([255.0 / 255, 255.0 / 255, 255.0 / 255])  # grey 116 now white
watchful_color_min = np.array([151.0 / 255, 138.0 / 255, 24.0 / 255])  # grey
watchful_color_max = np.array([151.0 / 255, 49.0 / 255, 151.0 / 255])  # orange
critical_color = np.array([151.0 / 255, 138.0 / 255, 24.0 / 255])  # orange
fatal_color = np.array([24.0 / 255, 24.0 / 255, 24.0 / 255])  # black
unknown_color = np.array([151.0 / 255, 49.0 / 255, 151.0 / 255])  # purple


def colortofloat(color):
    color = color.clip(0, 1)
    color = color * 255
    color = color.astype(np.int)
    res = ((color[0, :, :] << 16) | (color[1, :, :] << 8) | (color[2, :, :])).view(np.float32)
    return res


def convert_risk_to_colors(value, normals=None):
    # color = unknown_color
    if value < watchful_risk:
        color = safe_color
    if (value > watchful_risk) * (value < critical_risk):
        color = watchful_color_min + ((value - watchful_risk) / (critical_risk - watchful_risk)) * (
            watchful_color_max - watchful_color_min
        )
    if (value > critical_risk) * (value < fatal_risk):
        color = critical_color

    if value > fatal_risk:
        color = fatal_color

    return color.clip(0, 1)


# Create a colormap
num_colors = 256  # You can adjust the number of colors as needed
colormap_colors = [convert_risk_to_colors(value) for value in np.linspace(0, 1, num_colors)]
colormap_name = "traversability_risk"
racer_risk_colormap = mcolors.ListedColormap(colormap_colors, name=colormap_name)
racer_risk_colormap.set_bad(color=unknown_color)

if __name__ == "__main__":
    import time
    import os
    from perception_bev_learning.visu import LearningVisualizer
    from perception_bev_learning.cfg import ExperimentParams
    from perception_bev_learning.dataset import get_bev_dataloader

    from perception_bev_learning import BEV_ROOT_DIR

    cfg = ExperimentParams()
    cfg.update()
    v_min, v_max = 0, 1

    visu = LearningVisualizer(p_visu=os.path.join(BEV_ROOT_DIR, "results", "visu"), store=True)

    cfg.dataloader_train.num_workers = 0
    loader_train, loader_val, loader_test = get_bev_dataloader(cfg)
    import time

    st = time.time()
    for j, batch in enumerate(loader_train):
        imgs, rots, trans, intrins, post_rots, post_trans, target, aux, img_plots, gmr, pcd_new = batch

        print(j)
