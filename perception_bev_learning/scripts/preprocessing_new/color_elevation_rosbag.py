import rosbag
import numpy as np
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap, GridMapInfo
import torch

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

# Input and output bag file paths
input_bag_file = "/home/manthan/bev_cover/halter_ranch_cover_v2_2024-06-02-02-03-38.bag"
output_bag_file = "/home/manthan/bev_cover/cover_short_v2.bag"

def colortofloat(color):
    color = color.clip(0, 1)
    color = color * 255
    color = np.uint8(color)
    print(f"before color {color.shape}")
    print(color)
    res = np.float32((color[0, :, :] << 16) | (color[1, :, :] << 8) | (color[2, :, :]))
    print(res)
    return res

def colortofloattensor(color):
        color = color.clip(0, 1)
        color = color * 255
        color = color.type(torch.int)
        res = ((color[0, :, :] << 16) | (color[1, :, :] << 8) | (color[2, :, :])).view(torch.float32)
        return res

def gridmap(msg):
    data_in = msg[0]

    size_x = data_in["data"].shape[1]
    size_y = data_in["data"].shape[2]

    data_dim_0 = MultiArrayDimension()
    data_dim_0.label = "column_index"  # y dimension
    data_dim_0.size = size_y  # number of columns which is y
    data_dim_0.stride = size_y * size_x  # rows*cols
    data_dim_1 = MultiArrayDimension()
    data_dim_1.label = "row_index"  # x dimension
    data_dim_1.size = size_x  # number of rows which is x
    data_dim_1.stride = size_x  # number of rows
    layers = []
    data = []

    for i in range(data_in["data"].shape[0]):
        data_tmp = Float32MultiArray()
        data_tmp.layout.dim.append(data_dim_0)
        data_tmp.layout.dim.append(data_dim_1)
        data_tmp.data = data_in["data"][i, ::-1, ::-1].transpose().ravel()
        data.append(data_tmp)

    info = GridMapInfo()
    info.pose.position.x = data_in["position"][0]
    info.pose.position.y = data_in["position"][1]
    info.pose.position.z = data_in["position"][2]
    info.pose.orientation.x = data_in["orientation_xyzw"][0]
    info.pose.orientation.y = data_in["orientation_xyzw"][1]
    info.pose.orientation.z = data_in["orientation_xyzw"][2]
    info.pose.orientation.w = data_in["orientation_xyzw"][3]
    info.header = msg[1]
    # info.header.stamp.secs = msg[2]vis
    # info.header.stamp = rospy.Time.now()
    info.resolution = data_in["resolution"]
    info.length_x = size_x * data_in["resolution"]
    info.length_y = size_y * data_in["resolution"]

    gm_msg = GridMap(
        info=info,
        layers=data_in["layers"],
        basic_layers=data_in["basic_layers"],
        data=data,
    )

    return gm_msg

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

# Function to process gridmap message
def process_gridmap(msg):
    # Get the index of the elevation layer
    

    gridmap_data = convert_gridmap_float32(msg, ["elevation","elevation_color","cost"])

    np_data = gridmap_data["data"]

    for idx, layer in enumerate(gridmap_data["layers"]):
        print(layer)
        if layer == "elevation_color":
            elevation_idx = idx
        if layer == "cost":
            cost_idx = idx

    if elevation_idx is None:
        print("Elevation layer not found in the GridMap message")
        return msg

    # Get the elevation data
    elevation_data = np_data[elevation_idx]
    cost_data = np_data[cost_idx]
    # observed_color = np.array([0,1,0])  # light grey

    # print(elevation_data.shape)
    # elevation_data = np.repeat(elevation_data[None], 3, 0)
    # print(elevation_data.shape)
    # elevation_data[:,:,:] = observed_color[:,None, None]
    # print((elevation_data.shape))

    # colormap = np.array(colortofloattensor(torch.tensor(elevation_data)))
    # print((colormap.shape))
    # Clip the elevation data between -20 and 20
    clipped_elevation_data = np.clip(elevation_data, -20, 20)
    # Normalize the clipped elevation data between 0 and 1
    normalized_elevation_data = (clipped_elevation_data - (-20)) / (20 - (-20))
    # Create a colormap with Viridis color scheme
    colormap = plt.get_cmap('viridis')
    # Normalize the elevation data to the range [0, 1] for colormap
    norm = Normalize(vmin=0, vmax=1)


    cost_data = cost_data / 0.6
    cost_data[cost_data>1] = 1
    cost_colors = CMAP_TRAVERSABILITY(cost_data) * 255

    # Convert the normalized elevation data to colors using the colormap
    elevation_colors = (colormap(norm(normalized_elevation_data)) * 255)
    print(elevation_colors.shape)
    color = torch.tensor(elevation_colors).permute((2,0,1))
    color = color.type(torch.int)
    res = ((color[0, :, :] << 16) | (color[1, :, :] << 8) | (color[2, :, :])).view(torch.float32)
    print(res.shape)


    np_data[elevation_idx] = res

    color = torch.tensor(cost_colors).permute((2,0,1))
    color = color.type(torch.int)
    res = ((color[0, :, :] << 16) | (color[1, :, :] << 8) | (color[2, :, :])).view(torch.float32)
    np_data[cost_idx] = res

    gridmap_data["data"] = np_data

    gm_msg = gridmap((gridmap_data, msg.info.header))

    # Return the modified message
    return gm_msg

# Open input and output bag files
with rosbag.Bag(output_bag_file, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(input_bag_file).read_messages():
        print(f"processing msg {topic}")
        # If the message is of type GridMap
        if topic == '/crl_rzr/traversability_map/map_short_bev':
            # Process the gridmap message
            modified_msg = process_gridmap(msg)
            # Write the modified message to the output bag file
            outbag.write(topic, modified_msg, t)
        # else:
        #     # For other topics, simply write them to the output bag file
        #     outbag.write(topic, msg, t)

print("Processing completed. Output saved to", output_bag_file)