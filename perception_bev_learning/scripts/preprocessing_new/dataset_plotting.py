#!/usr/bin/env python

import rospy
import rosbag
import numpy as np
from grid_map_msgs.msg import GridMap
import torch
import numpy as np
from os.path import join
from mpl_toolkits.axes_grid1 import ImageGrid

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
from torchvision.transforms.functional import center_crop
import torch.nn.functional as F

CMAP_TRAVERSABILITY = sns.color_palette("RdYlBu_r", as_cmap=True)
CMAP_ELEVATION = sns.color_palette("viridis", as_cmap=True)
CMAP_ERROR = sns.color_palette("vlag", as_cmap=True)
CMAP_TRAVERSABILITY.set_bad(color="black")
CMAP_ELEVATION.set_bad(color="black")
CMAP_ERROR.set_bad(color="black")


def convert_gridmap_float32(msg, extract_layers):
    res = []
    layers_out = []
    for layer in extract_layers:
        if layer in msg.layers:
            # extract grid_map layer as numpy array
            layers_out.append(layer)
            data_list = msg.data[msg.layers.index(layer)].data
            layout_info = msg.data[msg.layers.index(layer)].layout
            assert layout_info.data_offset == 0
            assert layout_info.dim[1].stride == layout_info.dim[1].size
            assert layout_info.dim[0].label == "column_index"
            n_cols = layout_info.dim[0].size
            assert layout_info.dim[1].label == "row_index"
            n_rows = layout_info.dim[1].size
            data_in_layer = np.reshape(np.array(data_list), (n_rows, n_cols))
            data_in_layer = data_in_layer[::-1, ::-1].transpose().astype(np.float32)
            res.append(data_in_layer)
    out = {"data": np.stack(res)}
    out["layers"] = layers_out

    out["basic_layers"] = msg.basic_layers
    out["resolution"] = msg.info.resolution
    out["length"] = np.array([msg.info.length_x, msg.info.length_y])
    out["position"] = np.array(
        [msg.info.pose.position.x, msg.info.pose.position.y, msg.info.pose.position.z]
    )
    out["orientation_xyzw"] = np.array(
        [
            msg.info.pose.orientation.x,
            msg.info.pose.orientation.y,
            msg.info.pose.orientation.z,
            msg.info.pose.orientation.w,
        ]
    )
    return out
    
def read_gridmap_from_bag(bag_file, topic):
    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:

        costmap_exists = False
        trav_exists = False
        traj_exists = False
        
        traj_list = []
        trav_list = []
        costmap_list = []
    
        # Iterate through messages in the bag file
        for idx , (topic, msg, t) in enumerate(bag.read_messages(topics=topic)):
            

            # First Plot the Traversability Map
            out_trav = convert_gridmap_float32(msg, ["cost", "elevation_color"])
            # print(out_trav["data"].shape)
            # print(out_trav["layers"])

            color_maps = [CMAP_TRAVERSABILITY] * 1 + [CMAP_ELEVATION] * 1

            maps = out_trav["data"][:, ::-1, ::-1]

            v_mins = [0.0] * 1 + [-25.0] * 1
            v_maxs = [1.0] * 1 + [25] * 1

            N = maps.shape[0]
            nrows_ncols = (1, N)
            fig = plt.figure(figsize=(nrows_ncols[1] * 4.0 - 3, nrows_ncols[0] * 4.0))
            fig.tight_layout(pad=1.1)
            grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

            for i in range(N):

                v_min = (
                    maps[i][~np.isnan(maps[i])].min() if v_mins[i] is None else v_mins[i]
                )
                v_max = (
                    maps[i][~np.isnan(maps[i])].max() if v_maxs[i] is None else v_maxs[i]
                )

                grid[i].imshow(maps[i], cmap=color_maps[i], vmin=v_min, vmax=v_max)
                grid[i].grid(False)
            

            plt.savefig(f"/home/manthan/bev_dataset/sample_HR_{idx}.svg", format="svg", dpi=500)
    

if __name__ == '__main__':
    # Initialize ROS node
    # rospy.init_node('gridmap_bag_reader')

    # Get bag file path and topic name from ROS parameters
    bag_file = "/home/manthan/bev_dataset/halter_ranch_2024-03-26-03-56-28.bag"
    topic = ["/crl_rzr/traversability_map/map_short_bev"]

    # Read GridMap messages from bag file
    gridmap_array = read_gridmap_from_bag(bag_file, topic)