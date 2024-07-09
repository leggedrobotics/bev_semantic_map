from perception_bev_learning.cfg import ExperimentParams
from perception_bev_learning.dataset import get_bev_dataloader
from perception_bev_learning.visu import LearningVisualizer
from perception_bev_learning import BEV_ROOT_DIR
from perception_bev_learning.utils import denormalize_img

import numpy as np
from os.path import join

cfg = ExperimentParams()
cfg.update()

cfg.dataloader_train.batch_size = 1
cfg.dataloader_val.batch_size = 1
cfg.dataloader_test.batch_size = 1

loader_train, loader_val = get_bev_dataloader(cfg, return_test_dataloader=False)

visu = LearningVisualizer(
    p_visu=join(BEV_ROOT_DIR, "results/visu_validation_dataloader"), store=True, pl_model=None, log=False
)

for j, batch in enumerate(loader_val):
    imgs, rots, trans, intrins, post_rots, post_trans, target, aux, img_plots, gmr, pcd_new = batch
    b = 0
    img_idx_str = str(j)
    img_idx_str = "0" * (6 - len(img_idx_str)) + img_idx_str

    for k, l in enumerate(cfg.dataset_train.target_layers):
        if k == 0:
            v_min = 0
            v_max = 1
            elevation_map = False
        else:
            v_min = None
            v_max = None
            elevation_map = True

        visu.plot_elevation_map(
            target[b, k].clone(),
            elevation_map=elevation_map,
            v_min=v_min,
            v_max=v_max,
            tag=f"gt_{l.name}_{img_idx_str}",
        )
        out_imgs = []
        for c in cfg.visu.project_cams:
            out_imgs.append(visu.plot_image(np.array(denormalize_img(batch[0][b][c])), not_log=True, store=False))

        all_imgs = visu.plot_list(
            out_imgs,
            tag=f"imgs_{img_idx_str}",
        )
