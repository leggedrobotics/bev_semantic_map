import torch
from PIL import Image
import numpy as np
import os
from bevnet.dataset import normalize_img


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.nr_cameras = 4
        self.nr_lidar_points_time = 1

    def __len__(self):
        return 100

    def get_image_data(self):
        imgs = []
        img_plots = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for _ in range( self.nr_cameras):
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.eye(3)
            img = np.zeros((640,480,3), dtype=np.uint8)
            img_plot = normalize_img(img)
            
            # perform some augmentation on the image / is now ignored
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            
            imgs.append(normalize_img(img))
            intrins.append(intrin)
            
            H_base__camera = torch.eye(4)
            rots.append(H_base__camera[:3, :3])
            trans.append(H_base__camera[:3, 3])
            
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            img_plots.append(img_plot)

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
        for idx_pointcloud in range( self.nr_lidar_points_time ):
            
            points_in_base_frame = torch.rand((5000,3))
            pcd_new["points"].append(points_in_base_frame)
        return pcd_new
    
    def __getitem__(self, index):
        H_base__map = torch.eye(4)
        grid_map_resolution = torch.tensor( [0.2] )
        
        target, aux = torch.zeros((512,512,1)), torch.zeros((512,512,1))
        imgs, rots, trans, intrins, post_rots, post_trans, img_plots = self.get_image_data()
        pcd_new = self.get_raw_pcd_data()

        return (
            imgs,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            target,
            aux,
            img_plots,
            grid_map_resolution,
            pcd_new,
        )


def collate_fn(batch):
    output_batch = []
    for i in range(len(batch[0])):
        
        if type(batch[0][i]) != dict:
            # iterate over tuple of data
            output_batch.append(torch.stack([item[i] for item in batch]))
        else:
            # dicts are raw pointclouds
            res = {}
            stacked_scans_ls = []
            stacked_scan_indexes = []
            for j in range(len(batch)):
                stacked_scans_ls.append(torch.cat(batch[j][i]["points"]))
                stacked_scan_indexes.append(torch.tensor([scan.shape[0] for scan in batch[j][i]["points"]]))

            res["points"] = torch.cat(stacked_scans_ls)
            res["scan"] = torch.cat(stacked_scan_indexes)
            res["batch"] = torch.stack(stacked_scan_indexes).sum(1)

            output_batch.append(res)

    return tuple(output_batch)


def get_bev_dataloader(return_test_dataloader=True):
    dataset_train = DemoDataset()
    dataset_val = DemoDataset()

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4 , collate_fn=collate_fn)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4 , collate_fn=collate_fn)

    if return_test_dataloader:
        dataset_test = DemoDataset()
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, collate_fn=collate_fn)
        return loader_train, loader_val, loader_test
    
    return loader_train, loader_val


if __name__ == '__main__':
    loader_train, loader_val, loader_test = get_bev_dataloader()
    for j,batch in enumerate( loader_train):
        print(j)
        
        