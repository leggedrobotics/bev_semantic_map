import numpy as np
from .base import Converter, get_tf_and_header_dict
from typing import Any, Optional, Union, Tuple, List


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


class GridMapConverter(Converter):
    def __init__(
        self,
        reference_frame=None,
        aux_target_frame: Optional[List[str]] = None,
        layers: List[str] = None,
    ):
        super().__init__()
        self.ref_frame = reference_frame
        self.aux_target_frame = aux_target_frame
        self.layers = list(layers)

    def msg_type(self) -> Any:
        # TODO Fix this
        return "GridMap"

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener
    ) -> bool:
        fieldname = dataset_key
        res_dict = convert_gridmap_float32(msg, extract_layers=self.layers)

        # TODO: Generic version with auxilliary TFs
        tf_dict, suc = get_tf_and_header_dict(
            tf_listener, msg.info.header, self.ref_frame, msg.info.header.frame_id
        )

        if not suc:
            return False

        res_dict.update(tf_dict)
        static_keys = ["layers", "resolution", "length", "header_frame_id"]
        static_dict = {k: v for k, v in res_dict.items() if k in static_keys}
        dataset_writer.add_static(seq_name, fieldname, static_dict)
        dynamic_dict = {
            k: v
            for k, v in res_dict.items()
            if k not in static_keys and k != "basic_layers"
        }
        dataset_writer.add_data(seq_name, fieldname, dynamic_dict)

        return True
