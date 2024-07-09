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
CMAP_ELEVATION = sns.color_palette("viridis_r", as_cmap=True)
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

def marker_array_callback(msg):

    plt.figure()
    print(f"New msg")
    # Iterate through markers in the MarkerArray
    for marker in msg.markers:
        print(f"Marker New")
        # for point in marker.points:
        #     # Extract marker data
        #     x = point.x
        #     y = point.y
        #     z = point.z

        #     # Plot the marker as a point in 3D space
        #     plt.scatter(x, y, color='black', s=5)  # Use black color and smaller markers
        points = marker.points
        num_points = len(points)

        # Plot each line segment
        for idx, i in enumerate(range(0, num_points, 2)):
            
            if idx == 0:
                x_center, y_center = points[i].x * 2, points[i].y * 2
            
            x_values = [2* points[i].x - x_center - 200, 2* points[i+1].x - x_center - 200]
            y_values = [2* points[i].y - y_center - 200, 2* points[i+1].y - y_center - 200]
            x_values = [-x for x in x_values]
            y_values = [-y for y in y_values]

            plt.plot(y_values, x_values, color='black')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Top-Down View of MarkerArray')
    plt.xlim([0, 400])
    plt.ylim([0, 400])

    # Set aspect ratio to equal and turn off gridlines
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)

    # Save the plot as SVG
    plt.savefig('/home/manthan/marker_array.svg', format='svg')
    
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
            
            print(topic, t)
            if topic == "/crl_rzr/planner_short_forward/tree":
                traj_exists = True
                traj_list.append(msg)
            elif topic == "/crl_rzr/traversability_map/map_short":
                trav_exists = True
                trav_list.append(msg)
            elif topic == "/crl_rzr/planner_short/costmap_short":
                costmap_exists = True
                costmap_list.append(msg)

        # Close the bag file
        bag.close()

        # Ensure all topics have the same number of messages
        num_messages = min(len(traj_list), len(trav_list), len(costmap_list))

        # Iterate over the triplets of messages
        for idx in range(num_messages):
            
            trav = trav_list[idx]
            costmap = costmap_list[idx]
            traj = traj_list[idx]

            # First Plot the Traversability Map
            out_trav = convert_gridmap_float32(trav, ["cost"])
            # print(out_trav["data"].shape)
            # print(out_trav["layers"])

            out_costmap = convert_gridmap_float32(costmap, ["min"])
            # print(out_costmap["data"].shape)
            # print(out_costmap["layers"])


            color_maps = [CMAP_TRAVERSABILITY] * 1 + [CMAP_ELEVATION] * 1

            maps = np.vstack((out_trav["data"][:, ::-1, ::-1], out_costmap["data"][:, ::-1, ::-1]))

            cost_max = out_costmap["data"][0, 200, 200]

            v_mins = [0.0] * 2
            v_maxs = [1.0] * 1 + [cost_max] * 1

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
            

            for marker in traj.markers:
                print(f"Marker New")
                
            points = marker.points
            num_points = len(points)

            # Plot each line segment
            for c_idx, i in enumerate(range(0, num_points, 2)):
                
                if c_idx == 0:
                    x_center, y_center = points[i].x * 2, points[i].y * 2
                
                x_values = [2* points[i].x - x_center - 200, 2* points[i+1].x - x_center - 200]
                y_values = [2* points[i].y - y_center - 200, 2* points[i+1].y - y_center - 200]
                x_values = [-x for x in x_values]
                y_values = [-y for y in y_values]

                for i in range(N):
                    grid[i].plot(y_values, x_values, color='orange', linewidth=0.3)

            plt.savefig(f"/home/manthan/bev_planner/plots/sample_2_X_{idx}.svg", format="svg", dpi=500)
    

if __name__ == '__main__':
    # Initialize ROS node
    # rospy.init_node('gridmap_bag_reader')

    # Get bag file path and topic name from ROS parameters
    bag_file = "/home/manthan/bev_planner/sample_2_X_2024-03-23-06-20-02.bag"
    topic = ["/crl_rzr/planner_short_forward/tree", "/crl_rzr/traversability_map/map_short", "/crl_rzr/planner_short/costmap_short"]

    # Read GridMap messages from bag file
    gridmap_array = read_gridmap_from_bag(bag_file, topic)