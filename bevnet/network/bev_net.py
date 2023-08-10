import bevnet.network as network_simple
from bevnet.network import voxelize_pcd_scans
from torch import nn
import torch
from dataclasses import asdict
from bevnet.cfg import ModelParams

from bevnet import network


def get(name: str):
    for ele in dir(network):
        querry_name = name.replace("_", "")
        if ele.lower() == querry_name:
            return getattr(network, ele)

    raise ValueError(f"Could not find {name} in {dir(network)}")


class BevNet(nn.Module):
    def __init__(self, cfg_model: ModelParams):
        super(BevNet, self).__init__()
        self.cfg_model = cfg_model

        if cfg_model.dummy:
            self.dummy = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            return

        fusion_net_input_channels = 0

        if cfg_model.image_backbone != "skip":
            cfg = getattr(cfg_model, cfg_model.image_backbone)
            self.image_backbone = get(cfg_model.image_backbone)(cfg)
            fusion_net_input_channels += cfg.output_channels

        if cfg_model.pointcloud_backbone != "skip":
            cfg = getattr(cfg_model, cfg_model.pointcloud_backbone)
            self.pointcloud_backbone = get(cfg_model.pointcloud_backbone)(cfg)
            fusion_net_input_channels += cfg.output_channels

        if cfg_model.fusion_net.multi_head:
            self.fusion_net = network.MultiHeadBevEncode(
                fusion_net_input_channels, cfg_model.fusion_net.output_channels
            )
        else:
            self.fusion_net = network.BevEncode(fusion_net_input_channels, cfg_model.fusion_net.output_channels)

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, target_shape, pcd_new):
        """

        Args:
            imgs (torch.tensor shape=(BS, NR_CAMS, 3, H, W)): Camera Images
            rots (torch.tensor shape=(BS, NR_CAMS, 3, 3)): extrinsic camera rotation
            trans (torch.tensor shape=(BS, NR_CAMS, 3)): extrinsic camera translation
            intrins (torch.tensor shape=(BS, NR_CAMS, 3, 3)): intrinsic
            post_rots (torch.tensor shape=(BS, NR_CAMS, 3, 3)): transformations applied to pixel coordinates
            post_trans (torch.tensor shape=(BS, NR_CAMS, 3)): transformations applied to pixel coordinates
            target_shape (torch.tensor shape=4): indicates shape of output target (BS, OUT_DIMS, GRID_CELLS_X, GRID_CELLS_Y)
            pcd_new (dict): "points": (N,3) float32 ; "scan": (NR_TOTAL_SCANS) torch.int64 indicates where a new scan begins;  "batch": (NR_TOTAL_BATCHES) torch.int64 indicates to wich batch poins belong;


            --------------
            pcd_new format explained:  "scan"=[500,302,400,501] ; "batch"=[802,901] indicates the first scan is from point 0-500 second scan 500-802 ...
            same goes for the batches 0-802 is batch 0 therefore the first to scans belong to batch=0 and points 802-1703 to second batch.

        Returns:
            (torch.tensor shape=(BS, OUT_DIMS, GRID_CELLS_X, GRID_CELLS_Y)): shape of output target
        """

        if self.cfg_model.dummy:
            return self.dummy(
                voxelize_pcd_scans(
                    pcd_new["points"],
                    pcd_new["batch"],
                    pcd_new["scan"],
                    self.cfg_model.pointcloud_motion_net.gm_dim,
                    self.cfg_model.pointcloud_motion_net.gm_res,
                )[:, :, 0]
            ).clip(0, 1)

        features = []
        if hasattr(self, "pointcloud_backbone"):
            try:
                # print("pcd feat:", pcd_new["points"].shape, pcd_new["batch"], pcd_new["scan"])
                pcd_features = self.pointcloud_backbone(
                    x=pcd_new["points"], batch=pcd_new["batch"], scan=pcd_new["scan"]
                )
                # print("pcd feat:", pcd_features.shape)
                # print("target shape:", target_shape)
                pcd_features = torch.nn.functional.interpolate(pcd_features, size=(target_shape[2], target_shape[3]))
                print("pcd feat:", pcd_features.shape)
                features.append(pcd_features)
            except Exception as e:
                raise ValueError("Pointcloud backbone failed")
        if hasattr(self, "image_backbone"):
            camera_info = {
                "rots": rots,
                "trans": trans,
                "intrins": intrins,
                "post_rots": post_rots,
                "post_trans": post_trans,
            }
            all_data = {"camera_info": camera_info, "imgs": imgs, "pcd_new": pcd_new}

            image_features = self.image_backbone(
                imgs, rots, trans, intrins, post_rots, post_trans, pcd_new=pcd_new, camera_info=camera_info
            )
            # print("image feat:", image_features.shape)
            features.append(image_features)

        features = torch.cat(features, dim=1)
        return self.fusion_net(features).contiguous()   # Store the tensor in a contiguous chunk of memory for efficiency


if __name__ == "__main__":

    from bevnet.dataset import get_bev_dataloader

    cfg = ModelParams()
    model = BevNet(cfg)
    model.cuda()

    loader_train, loader_val, loader_test = get_bev_dataloader()
    for j, batch in enumerate(loader_train):
        print(j)
        imgs, rots, trans, intrins, post_rots, post_trans, target, *_, pcd_new = batch
        pcd_new["points"], pcd_new["batch"], pcd_new["scan"] = pcd_new["points"].cuda(), pcd_new["batch"].cuda(), pcd_new["scan"].cuda()
        pred = model(imgs.cuda(), rots.cuda(), trans.cuda(), intrins.cuda(), post_rots.cuda(), post_trans.cuda(), target.cuda().shape, pcd_new)

    print("pred:", pred.shape)
    print(pred)
