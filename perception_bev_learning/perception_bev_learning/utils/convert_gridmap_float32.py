import numpy as np
from perception_bev_learning.dataset.h5py_keys import TRAV_LAYERS


def convert_gridmap_float32(msg, only_header=False):
    if not only_header:
        res = []
        layers_out = []
        for layer in TRAV_LAYERS:
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
    else:
        out = {}

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
