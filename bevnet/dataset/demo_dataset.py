import torch
from PIL import Image
import numpy as np
import os
from bevnet.dataset import normalize_img
from bevnet.cfg import DataParams


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data: DataParams):
        super(DemoDataset, self).__init__()
        self.cfg_data = cfg_data

    def __len__(self):
        return self.cfg_data.nr_data

    def get_image_data(self):
        imgs = []
        img_plots = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for _ in range(self.cfg_data.nr_cameras):
            # post_rot = torch.eye(2)
            # post_tran = torch.zeros(2)
            intrin = torch.eye(3)
            img = np.zeros((self.cfg_data.img_width, self.cfg_data.img_height, 3), dtype=np.uint8)
            img_plot = normalize_img(img)  # Perform potential augmentation to plot the image

            # perform some augmentation on the image / is now ignored
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)

            imgs.append(normalize_img(img))
            intrins.append(intrin)

            H_base__camera = torch.eye(4)  # 4d tensor for tf from base to camera frame
            rots.append(H_base__camera[:3, :3])
            trans.append(H_base__camera[:3, 3])

            post_rots.append(post_rot)
            post_trans.append(post_tran)  # Rotation after performing augmentation
            img_plots.append(img_plot)  # Translation after performing augmentation

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
            torch.stack(img_plots),
        )

    def get_raw_pcd_data(self):
        pcd_new = {}
        pcd_new["points"] = []
        # for idx_pointcloud in range(self.cfg_data.nr_lidar_points_time): # Only one lidar point for now
        for idx_pointcloud in range(self.cfg_data.nr_lidar_points_time):  # Only one lidar point for now

            points_in_base_frame = torch.rand((self.cfg_data.nr_points, 3))  # Random points, uniformly between 0 and 1
            pcd_new["points"].append(points_in_base_frame)
        return pcd_new

    def __getitem__(self, index):  # Called when iterating over the dataset
        H_base__map = torch.eye(4)  # 2d tf matrix from base to map frame
        grid_map_resolution = torch.tensor([self.cfg_data.gird_map_resolution])

        # target, aux = torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512))    # Labels and aux labels in BEV space
        target, aux = (
            torch.zeros(self.cfg_data.target_shape),
            torch.zeros(self.cfg_data.aux_shape),
        )  # Labels and aux labels in BEV space
        imgs, rots, trans, intrins, post_rots, post_trans, img_plots = self.get_image_data()
        pcd_new = self.get_raw_pcd_data()

        return (
            imgs,
            rots,
            trans,
            intrins,
            post_rots,  # After performing augmentations
            post_trans,  # After performing augmentations
            target,
            aux,  # aux is ignored
            img_plots,  # img_plots is ignored
            grid_map_resolution,  # grid_map_resolution is ignored
            pcd_new,
        )


def collate_fn(batch):  # Prevents automatic data loading, performs operations over batches of data
    output_batch = []
    for i in range(len(batch[0])):

        if type(batch[0][i]) != dict:
            # iterate over tuple of data
            output_batch.append(torch.stack([item[i] for item in batch]))
        else:
            # dicts are raw pointclouds, only perform this for the pointclouds
            res = {}
            stacked_scans_ls = []
            stacked_scan_indexes = []
            for j in range(len(batch)):
                stacked_scans_ls.append(torch.cat(batch[j][i]["points"]))  # Concatenate all scans
                stacked_scan_indexes.append(
                    torch.tensor([scan.shape[0] for scan in batch[j][i]["points"]])
                )  # Get the number of points in each scan

            res["points"] = torch.cat(stacked_scans_ls)
            res["scan"] = torch.cat(stacked_scan_indexes)
            res["batch"] = torch.stack(stacked_scan_indexes).sum(1)  # Get the number of points in each scan

            output_batch.append(res)

    return tuple(output_batch)


def get_bev_dataloader(return_test_dataloader=True):

    data_cfg = DataParams()

    dataset_train = DemoDataset(data_cfg)
    dataset_val = DemoDataset(data_cfg)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=collate_fn)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, collate_fn=collate_fn)

    if return_test_dataloader:
        dataset_test = DemoDataset(data_cfg)  # Create a new test dataset with random values
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn)
        return loader_train, loader_val, loader_test

    return loader_train, loader_val


if __name__ == "__main__":
    loader_train, loader_val, loader_test = get_bev_dataloader()
    for j, batch in enumerate(loader_train):
        print(j)
