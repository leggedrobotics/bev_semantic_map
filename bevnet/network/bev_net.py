from bevnet.network import voxelize_pcd_scans
import torch
from bevnet.cfg import ModelParams
from icecream import ic

from bevnet import network

import torch.nn.functional as F
import torchshow as ts

# Global settings
SAVE_PRED = True


def get(name: str):
    for ele in dir(network):
        querry_name = name.replace("_", "")
        if ele.lower() == querry_name:
            return getattr(network, ele)

    raise ValueError(f"Could not find {name} in {dir(network)}")


class BevNet(torch.nn.Module):
    def __init__(self, cfg_model: ModelParams):
        super(BevNet, self).__init__()
        self.cfg_model = cfg_model

        # Setup model structure
        if cfg_model.dummy:
            self.dummy = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
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
            print("Using MultiHeadBevEncode")
            self.fusion_net = network.MultiHeadBevEncode(
                fusion_net_input_channels, cfg_model.fusion_net.output_channels
            )
        if cfg_model.fusion_net.anomaly:
            print("Using AnomalyBevEncode")
            # fusion_net_input_channels = 160
            self.fusion_net = network.LinearRNVP(
                input_dim=fusion_net_input_channels,
                coupling_topology=[200],
                flow_n=20,
                batch_norm=False,
                mask_type="odds",
                conditioning_size=0,
                use_permutation=True,
                single_function=True,
            )
        if cfg_model.fusion_net.simple_mlp:
            print("Using SimpleMLP")
            self.fusion_net = network.SimpleMLP(
                input_size=fusion_net_input_channels, hidden_sizes=[256, 32, 1], reconstruction=False
            )
        # else:
        #     self.fusion_net = network.BevEncode(fusion_net_input_channels, cfg_model.fusion_net.output_channels)

        # self.optimizer = torch.optim.Adam(self.fusion_net.parameters(), lr=cfg_model.fusion_net.lr)
        # self.loss = torch.nn.MSELoss()

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, target_shape, pcd_new, target=None):
        """

        Args:
            imgs (torch.tensor shape=(BS, NR_CAMS, 3, H, W)): Camera Images
            rots (torch.tensor shape=(BS, NR_CAMS, 3, 3)): extrinsic camera rotation
            trans (torch.tensor shape=(BS, NR_CAMS, 3)): extrinsic camera translation
            intrins (torch.tensor shape=(BS, NR_CAMS, 3, 3)): intrinsic
            post_rots (torch.tensor shape=(BS, NR_CAMS, 3, 3)): transformations applied to pixel coordinates
            post_trans (torch.tensor shape=(BS, NR_CAMS, 3)): transformations applied to pixel coordinates
            target_shape (torch.tensor shape=4): indicates shape of
             output target (BS, OUT_DIMS, GRID_CELLS_X, GRID_CELLS_Y)
            pcd_new (dict): "points": (N,3) float32 ; "scan": (NR_TOTAL_SCANS) torch.int64 indicates where a new scan
            begins;  "batch": (NR_TOTAL_BATCHES) torch.int64 indicates to which batch points belong;


            --------------
            pcd_new format explained:  "scan"=[500,302,400,501] ; "batch"=[802,901] indicates the first scan
             is from point 0-500 second scan 500-802 ...
            same goes for the batches 0-802 is batch 0 therefore the first two scans
            belong to batch=0 and points 802-1703 to second batch.

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
                # Change x, y, z to y, x, z
                pcd_new["points"] = pcd_new["points"][:, [1, 0, 2]]

                pcd_features = self.pointcloud_backbone(
                    x=pcd_new["points"], batch=pcd_new["batch"], scan=pcd_new["scan"]
                )
                pcd_features = torch.nn.functional.interpolate(pcd_features, size=(target_shape[2], target_shape[3]))
                # ts.show(pcd_features[0, :25, :, :])
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
            # flip x to minus x and y to minus y
            image_features = torch.flip(image_features, dims=(2, 3))
            # ts.show(image_features[0, :25, :, :])
            features.append(image_features)

        features = torch.cat(features, dim=1)  # Simply stack features from different backbones

        # ts.show(features[0, :25, :, :])
        # print("features shape:", features.shape)

        if self.cfg_model.fusion_net.anomaly:
            # Change feature dimension
            features = features.permute(0, 2, 3, 1)  # (BS, C, H, W) -> (BS, H, W, C)
            # features = features.view(-1, features.shape[-1])    # (BS, H, W, C) -> (BS*H*W, C)

            features = features.view(
                -1, features.shape[1] * features.shape[2], features.shape[-1]
            )  # (BS, H, W, C) -> (BS, H*W, C)

            # If target is available, mask out only positive samples
            if target is not None:
                # target = target.view(-1)
                target = target.view(-1, target.shape[2] * target.shape[3])  # (BS, H, W, C) -> (BS, H*W)
                features = features[target]
            else:
                features = features.view(-1, features.shape[-1])  # (BS, H, W, C) -> (BS*H*W, C)

        # return self.fusion_net(features).contiguous()  # Store the tensor in a contiguous chunk of memory for
        # efficiency

        # print(features.shape)
        # features = features.reshape(-1, features.shape[1])    # Only need this for 1d linear MLP
        # print(features.shape)
        return self.fusion_net(features)  # Store the tensor in a contiguous chunk of memory for efficiency


if __name__ == "__main__":

    model_cfg = ModelParams()
    model = BevNet(model_cfg)
    model.cuda()
